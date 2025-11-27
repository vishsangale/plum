import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalEncoder(nn.Module):
    """
    Encodes and fuses multiple input embeddings into a single vector.
    
    As described in the PLUM paper (Section 2.1.1), this module:
    1. Takes a list of embedding vectors {x_m} (e.g., text, video, audio).
    2. Encodes each x_m into a latent vector z_m using a separate encoder E_m.
    3. Concatenates them: z_tilde = [z_1, ..., z_M].
    4. Projects the concatenation to a unified vector z.
    """
    def __init__(self, input_dims: list[int], latent_dim: int, output_dim: int):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            ) for dim in input_dims
        ])
        
        # Projection layer after concatenation
        # Input size is M * latent_dim
        self.projector = nn.Sequential(
            nn.Linear(len(input_dims) * latent_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: List of tensors, where inputs[i] has shape (batch_size, input_dims[i])
        Returns:
            Fused embedding tensor of shape (batch_size, output_dim)
        """
        if len(inputs) != len(self.encoders):
            raise ValueError(f"Expected {len(self.encoders)} inputs, got {len(inputs)}")
            
        # Encode each modality
        latents = []
        for i, x in enumerate(inputs):
            z_m = self.encoders[i](x)
            latents.append(z_m)
            
        # Concatenate
        z_tilde = torch.cat(latents, dim=-1)
        
        # Project
        z = self.projector(z_tilde)
        return F.normalize(z, p=2, dim=-1)

class RQVAE(nn.Module):
    """
    Residual Quantized Variational AutoEncoder (RQ-VAE) with:
    1. Multi-Resolution Codebooks (Section 2.1.2)
    2. Progressive Masking (Section 2.1.2)
    """
    def __init__(self, input_dim: int, num_levels: int = 4, base_codebook_size: int = 2048):
        super().__init__()
        self.num_levels = num_levels
        self.input_dim = input_dim
        
        # Multi-resolution codebooks
        # Level 1: 2048, Level 2: 1024, Level 3: 512, etc.
        self.codebooks = nn.ModuleList()
        self.codebook_sizes = []
        
        for l in range(num_levels):
            # Formula: 2048 / 2^(l) -> Note: Paper says 2048 / 2^(level-1) where level starts at 1
            # So for l=0 (level 1), size = 2048 / 2^0 = 2048
            size = int(base_codebook_size / (2 ** l))
            self.codebook_sizes.append(size)
            
            # Each codebook maps indices to vectors of dimension input_dim
            # We use an Embedding layer for this
            self.codebooks.append(nn.Embedding(size, input_dim))
            
    def forward(self, z: torch.Tensor, training: bool = True, commitment_beta: float = 0.25, dropout_prob: float = 0.0, enable_progressive_masking: bool = True):
        """
        Args:
            z: Input tensor of shape (batch_size, input_dim)
            training: Whether in training mode (for progressive masking)
            enable_progressive_masking: Whether to apply progressive masking (random depth r)
            
        Returns:
            z_q: Quantized vector (sum of codes)
            codes: Indices of chosen codes at each level (batch_size, num_levels)
            commitment_loss: Loss to train codebooks
        """
        batch_size = z.shape[0]
        residual = z
        z_q = torch.zeros_like(z)
        codes = []
        commitment_loss = 0.0
        
        # Progressive Masking: Select a random depth r in [1, L]
        if training and enable_progressive_masking:
            r = torch.randint(1, self.num_levels + 1, (1,)).item()
        else:
            r = self.num_levels

            
        for l in range(self.num_levels):
            # If we are beyond the random depth r, we stop accumulating z_q
            # But we might still want to compute codes/loss for full training?
            # The paper says "mask is applied to select the first r codebook levels in SID training"
            # This implies we only use the first r levels for the forward pass reconstruction.
            
            # Find nearest neighbor in codebook
            # (batch_size, 1, dim) - (1, codebook_size, dim) -> (batch_size, codebook_size, dim)
            # Distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
            
            codebook = self.codebooks[l]
            # codebook.weight: (size, dim)
            
            # Efficient distance computation
            d = torch.sum(residual ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook.weight ** 2, dim=1) - \
                2 * torch.matmul(residual, codebook.weight.t())
            
            # Get indices
            min_encoding_indices = torch.argmin(d, dim=1) # (batch_size,)
            codes.append(min_encoding_indices)
            
            # Get quantized vectors
            z_e = codebook(min_encoding_indices) # (batch_size, dim)
            
            # Commitment loss: ||sg[z] - e||^2 + beta * ||z - sg[e]||^2
            # In RQ-VAE usually: ||residual - sg[z_e]||^2 + beta * ||sg[residual] - z_e||^2
            # Paper formula: beta * ||r_l - sg[e_l]||^2 + ||sg[r_l] - e_l||^2
            # Here r_l is the residual *before* this level.
            
            # Paper formula (Eq 2): beta * ||r_l - sg[e_l]||^2 + ||sg[r_l] - e_l||^2
            # Term 1: Encoder update (Commitment loss) -> beta * ||residual - sg[z_e]||^2
            # Term 2: Codebook update -> ||sg[residual] - z_e||^2
            
            # beta = 0.25  # Restored for codebook stability
            term1 = commitment_beta * torch.mean((residual - z_e.detach()) ** 2)
            term2 = torch.mean((residual.detach() - z_e) ** 2)
            commitment_loss += term1 + term2
            
            # Level Dropout Logic
            if training and l == 0 and dropout_prob > 0 and torch.rand(1).item() < dropout_prob:
                # Dropped: Do NOT use z_e for residual update or z_q accumulation
                # We still computed commitment loss above, so L0 is trained to match z
                pass
            else:
                # Not Dropped: Update residual and accumulate z_q
                residual = residual - z_e
                if l < r:
                    z_q = z_q + z_e
                
        # Straight-through estimator for backprop
        # z_q = z + (z_q - z).detach() 
        # But here z_q is a sum of multiple discrete selections.
        # Standard VQ-VAE trick: z_q = z + (z_q - z).detach() makes gradients flow from z_q to z.
        # In RQ-VAE, we want gradients to flow through the residuals.
        # The reconstruction loss will drive z_q to be close to z.
        # The paper doesn't explicitly mention the straight-through trick for the sum, 
        # but it's standard. Let's apply it to the final result.
        
        z_q = z + (z_q - z).detach()
        
        return z_q, torch.stack(codes, dim=1), commitment_loss

class PLUM_SID(nn.Module):
    """
    Main PLUM Semantic ID Model.
    Combines MultiModalEncoder, RQVAE, and Decoders.
    """
    def __init__(self, input_dims: list[int], latent_dim: int, output_dim: int, num_levels: int = 4, base_codebook_size: int = 2048):
        super().__init__()
        self.encoder = MultiModalEncoder(input_dims, latent_dim, output_dim)
        self.rqvae = RQVAE(output_dim, num_levels, base_codebook_size)
        
        # Decoders to reconstruct original embeddings from quantized vector z_q
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, dim)
            ) for dim in input_dims
        ])
        
    def forward(self, inputs: list[torch.Tensor], training: bool = True, commitment_beta: float = 0.25, dropout_prob: float = 0.0, enable_progressive_masking: bool = True):
        # 1. Encode and Fuse
        z = self.encoder(inputs)
        
        # 2. Quantize
        z_q, codes, commitment_loss = self.rqvae(z, training=training, commitment_beta=commitment_beta, dropout_prob=dropout_prob, enable_progressive_masking=enable_progressive_masking)

        
        # 3. Reconstruct
        reconstructions = []
        for decoder in self.decoders:
            reconstructions.append(decoder(z_q))
            
        return reconstructions, z_q, codes, commitment_loss

class ContrastiveLoss(nn.Module):
    """
    Co-occurrence Contrastive Loss (Section 2.1.3)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_q: torch.Tensor, z_q_pos: torch.Tensor):
        """
        Args:
            z_q: Quantized vectors for anchor items (batch_size, dim)
            z_q_pos: Quantized vectors for co-occurring positive items (batch_size, dim)
        """
        # Normalize vectors for cosine similarity (if dot product is intended, remove normalization)
        # Paper says "dot-product similarity", but usually contrastive loss uses cosine or normalized dot product.
        # Let's assume raw dot product as per paper text "dot-product similarity".
        # But standard InfoNCE usually benefits from temperature scaling.
        
        batch_size = z_q.shape[0]
        
        # Normalize vectors to unit length for cosine similarity
        # This is critical! Without normalization, dot products are unbounded
        z_q_norm = F.normalize(z_q, p=2, dim=1)
        z_q_pos_norm = F.normalize(z_q_pos, p=2, dim=1)
        
        # Similarity matrix: (batch_size, batch_size)
        # Now this is cosine similarity (normalized dot product)
        sim_matrix = torch.matmul(z_q_norm, z_q_pos_norm.t())
        
        # Apply temperature scaling and compute InfoNCE loss
        logits = sim_matrix / self.temperature
        labels = torch.arange(batch_size, device=z_q.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


