import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from plum.sid_model import PLUM_SID, ContrastiveLoss
from plum.data import SyntheticSIDDataset, PLUMSIDDataset
from plum.config import PLUMConfig
import os

import argparse
import torch.nn.functional as F
from tqdm import tqdm
import random

def train_sid():
    parser = argparse.ArgumentParser(description="Train PLUM SID Model")
    parser.add_argument("--dataset", type=str, default="movielens-1m", help="Dataset name (e.g., movielens-1m, movielens-10m)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps per epoch")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    config = PLUMConfig(dataset_name=args.dataset)
    if args.epochs:
        config.active_model_config.epochs = args.epochs
    if args.batch_size:
        config.active_model_config.batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {config.dataset_name}")
    
    # Print Ablation Settings
    print(f"Ablations: Contrastive={config.ablation.enable_contrastive_loss}, "
          f"ProgMask={config.ablation.enable_progressive_masking}, "
          f"DeadCode={config.ablation.enable_dead_code_revival}")

    # Load Data
    dataset = PLUMSIDDataset(config.active_dataset)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.active_model_config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    # Initialize Model
    sid_model = PLUM_SID(
        input_dims=config.active_dataset.input_dims,
        latent_dim=config.active_model_config.latent_dim,
        output_dim=config.active_model_config.output_dim,
        codebook_sizes=config.active_model_config.codebook_sizes,
        kmeans_init=config.active_model_config.kmeans_init
    ).to(device)
    
    # Initialize Optimizer
    optimizer = optim.Adam(sid_model.parameters(), lr=config.active_model_config.learning_rate)
    
    # Initialize Contrastive Loss
    contrastive_loss_fn = ContrastiveLoss(temperature=config.active_model_config.contrastive_temperature).to(device)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{config.dataset_name}_sid")
    
    # K-Means Initialization
    if config.active_model_config.kmeans_init:
        print("Initializing codebooks with K-Means...")
        # Get one large batch or accumulate
        # For simplicity, let's just take the first batch
        try:
            batch = next(iter(dataloader))
            emb1, _ = batch
            # emb1 is list of tensors. Move to device.
            emb1 = [e.to(device) for e in emb1]
            
            with torch.no_grad():
                z = sid_model.encoder(emb1)
                sid_model.rqvae.init_codebook(z)
        except StopIteration:
            print("Warning: Dataloader empty, skipping K-Means init.")

    print("Starting SID Training...")
    global_step = 0
    
    for epoch in range(config.active_model_config.epochs):
        sid_model.train()
        total_loss = 0
        total_recon_loss = 0
        total_commit_loss = 0
        total_contrastive_loss = 0
        
        # Track code usage
        epoch_unique_codes = [set() for _ in range(config.active_model_config.num_levels)]
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.active_model_config.epochs}")
        
        for batch_idx, (emb1, emb2) in enumerate(progress_bar):
            if args.max_steps and batch_idx >= args.max_steps:
                break
            emb1 = [e.to(device) for e in emb1]
            emb2 = [e.to(device) for e in emb2]
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # 1. Reconstruction (RQ-VAE) on emb1
            # Apply progressive masking if enabled
            # Note: RQVAE handles random depth selection internally if enable_progressive_masking=True
            recon_x, v_q, codes, commit_loss = sid_model(
                emb1, 
                training=True,
                commitment_beta=config.active_model_config.commitment_beta,
                dropout_prob=config.active_model_config.level_dropout_prob,
                enable_progressive_masking=config.ablation.enable_progressive_masking
            )
            
            # Track code usage (per level)
            for l in range(codes.shape[1]):
                epoch_unique_codes[l].update(codes[:, l].tolist())
            
            # Reconstruction Loss (MSE)
            recon_loss = 0
            for r_x, e in zip(recon_x, emb1):
                recon_loss += F.mse_loss(r_x, e)
            
            # 2. Contrastive Loss & Cosine Similarity
            cos_sim = 0.0
            if config.ablation.enable_contrastive_loss:
                # Forward pass for emb2
                _, v_q2, _, _ = sid_model(
                    emb2,
                    training=True,
                    commitment_beta=config.active_model_config.commitment_beta,
                    dropout_prob=0.0, # No dropout for target
                    enable_progressive_masking=False # No masking for target
                ) 
                contrast_loss = contrastive_loss_fn(v_q, v_q2)
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(v_q, v_q2).mean().item()
            else:
                contrast_loss = torch.tensor(0.0, device=device)
            
            # Total Loss
            loss = (config.active_model_config.recon_weight * recon_loss) + \
                   commit_loss + \
                   (config.active_model_config.contrastive_weight * contrast_loss)
                   
            loss.backward()
            optimizer.step()
            
            # --- Dead Code Revival ---
            if config.ablation.enable_dead_code_revival and batch_idx > 0 and batch_idx % 1000 == 0:
                with torch.no_grad():
                    for level in range(config.active_model_config.num_levels):
                        codebook_size = config.active_model_config.codebook_sizes[level]
                        used_codes = epoch_unique_codes[level]
                        unused_codes = set(range(codebook_size)) - used_codes
                        
                        if len(unused_codes) > 0:
                            # Reset unused codes to random embeddings from current batch
                            # Get some random embeddings from the current batch (emb1 is list of tensors)
                            # We use the first modality for simplicity, or fused z if accessible.
                            # Let's use the encoder to get z from the current batch
                            z_batch = sid_model.encoder(emb1) # (batch_size, output_dim)
                            
                            revived_indices = []
                            # Revive up to batch_size codes
                            n_to_revive = min(len(unused_codes), len(z_batch))
                            
                            for i, unused_idx in enumerate(list(unused_codes)[:n_to_revive]):
                                # Assign random z from batch
                                random_idx = torch.randint(0, len(z_batch), (1,)).item()
                                sid_model.rqvae.codebooks[level].weight.data[unused_idx] = z_batch[random_idx]
                                revived_indices.append(unused_idx)
                            
                            # Mark revived codes as used so they aren't reset again immediately
                            epoch_unique_codes[level].update(revived_indices)
                            
                            if batch_idx % 500 == 0:
                                print(f"  [Revival] Reset {len(revived_indices)} unused codes at level {level}")

            # Logging
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
            total_contrastive_loss += contrast_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(), 
                'recon': recon_loss.item(),
                'commit': commit_loss.item(),
                'contrast': contrast_loss.item(),
                'sim': cos_sim
            })
            
            writer.add_scalar('SID/Loss', loss.item(), global_step)
            writer.add_scalar('SID/Recon_Loss', recon_loss.item(), global_step)
            writer.add_scalar('SID/Commit_Loss', commit_loss.item(), global_step)
            writer.add_scalar('SID/Contrast_Loss', contrast_loss.item(), global_step)
            writer.add_scalar('SID/Cosine_Sim', cos_sim, global_step)
            global_step += 1
            
        # End of Epoch Stats
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_commit = total_commit_loss / len(dataloader)
        avg_contrast = total_contrastive_loss / len(dataloader)
        
        # Calculate Cosine Similarity for monitoring
        with torch.no_grad():
             cos_sim = F.cosine_similarity(v_q, v_q2).mean().item() if config.ablation.enable_contrastive_loss else 0.0
        
        total_unique = sum(len(s) for s in epoch_unique_codes)
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
              f"Total: {avg_loss:.4f}, Recon: {avg_recon:.4f} (w={config.active_model_config.recon_weight}), "
              f"Commit: {avg_commit:.4f}, Contrast: {avg_contrast:.4f} (w={config.active_model_config.contrastive_weight}), "
              f"CosSim: {cos_sim:.4f}, "
              f"Unique (Epoch): {total_unique}")
              
        writer.add_scalar('SID/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('SID/Unique_Codes', total_unique, epoch)
             
        # Save Checkpoint
        os.makedirs(config.active_dataset.checkpoint_dir, exist_ok=True)
        torch.save(sid_model.state_dict(), os.path.join(config.active_dataset.checkpoint_dir, f"sid_model_epoch_{epoch+1}.pth"))
        
    # Save Final Model
    final_path = os.path.join(config.active_dataset.checkpoint_dir, config.sid_model_checkpoint)
    torch.save(sid_model.state_dict(), final_path)
    print(f"Saved SID model to {final_path}")


if __name__ == "__main__":
    train_sid()
