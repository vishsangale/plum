import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from plum.sid_model import PLUM_SID, ContrastiveLoss
from plum.data import SyntheticSIDDataset, PLUMSIDDataset
from plum.config import PLUMConfig
import os

def train_sid():
    # Hyperparameters
    config = PLUMConfig()
    
    # Setup TensorBoard
    writer = SummaryWriter('runs/sid_training')
    
    # Setup
    # dataset = SyntheticSIDDataset(num_samples=1000, input_dims=config.active_dataset.input_dims)
    dataset = PLUMSIDDataset(config.active_dataset)
    dataloader = DataLoader(dataset, batch_size=config.active_model_config.batch_size, shuffle=True)
    
    model = PLUM_SID(
        config.active_dataset.input_dims, 
        config.active_model_config.latent_dim, 
        config.active_model_config.output_dim, 
        config.active_model_config.codebook_sizes,
        kmeans_init=config.active_model_config.kmeans_init
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.active_model_config.learning_rate)
    
    # K-means Initialization (if enabled)
    # We need to run one forward pass with a batch to initialize
    if config.active_model_config.kmeans_init:
        print("Running K-means initialization on first batch...")
        # Get one batch
        first_batch = next(iter(dataloader))
        # first_batch is (anchor_embeddings, positive_embeddings)
        # anchor_embeddings is a list of tensors [tensor(batch, 384)]
        anchor_embeddings = first_batch[0][0].to(device)
        
        # Run encoder to get z
        with torch.no_grad():
            z = model.encoder([anchor_embeddings])
            model.rqvae.init_codebook(z)
    criterion = ContrastiveLoss(temperature=config.active_model_config.contrastive_temperature)
    print(f"Contrastive Loss Temperature: {criterion.temperature}")
    
    print("Starting SID Training...")
    print("Run 'tensorboard --logdir=runs' to view training progress")
    
    global_step = 0
    
    # Track cumulative unique codes across all epochs for dead code revival
    all_time_unique_codes = [set() for _ in range(config.active_model_config.num_levels)]
    
    try:
        for epoch in range(config.active_model_config.epochs):
            total_loss = 0
            total_recon_loss = 0
            total_commitment_loss = 0
            total_contrastive_loss = 0
            total_cosine_sim = 0
            epoch_unique_codes = [set() for _ in range(config.active_model_config.num_levels)]
            model.train()
            # Constant contrastive weight from config
            contrastive_weight = config.active_model_config.contrastive_weight
    
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config.active_model_config.epochs} - Contrastive Weight: {contrastive_weight}")
            print(f"{'='*60}\n")
            
            for batch_idx, (anchor_embeddings, positive_embeddings) in enumerate(dataloader):
                # Move to device and L2 normalize
                anchor_embeddings = [torch.nn.functional.normalize(emb.to(device), p=2, dim=-1) for emb in anchor_embeddings]
                positive_embeddings = [torch.nn.functional.normalize(emb.to(device), p=2, dim=-1) for emb in positive_embeddings]
                
                # Forward pass for anchor
                reconstructions, z_q, codes, commitment_loss = model(
                    anchor_embeddings, 
                    commitment_beta=config.active_model_config.commitment_beta,
                    dropout_prob=config.active_model_config.level_dropout_prob,
                    enable_progressive_masking=config.active_model_config.enable_progressive_masking
                )
                
                # Track unique codes per level
                for level in range(config.active_model_config.num_levels):
                    unique_codes_in_batch = torch.unique(codes[:, level]).cpu().tolist()
                    epoch_unique_codes[level].update(unique_codes_in_batch)
                    all_time_unique_codes[level].update(unique_codes_in_batch)
                
                # Check total unique codes across all levels (using all-time usage)
                total_unique_codes = sum(len(codes_set) for codes_set in all_time_unique_codes)
                epoch_total_unique = sum(len(codes_set) for codes_set in epoch_unique_codes)
                
                # Safety check: Stop if codebook collapses
                # Threshold: 20% of total codebook capacity
                total_capacity = sum(model.rqvae.codebook_sizes)
                if batch_idx > 1000 and total_unique_codes < (0.2 * total_capacity):
                    print(f"\n⚠️  STOPPING: Codebook collapse detected! Unique codes: {total_unique_codes} < {0.2 * total_capacity}")
                    print(f"Contrastive weight {contrastive_weight} is too high. Reduce it or enable more aggressive code reset.")
                    return
                
                # Fixed reconstruction weight from config
                recon_weight = config.active_model_config.recon_weight
                
                # Reconstruction loss
                recon_loss = sum([torch.mean((emb - recon) ** 2) for emb, recon in zip(anchor_embeddings, reconstructions)])
                
                # Cosine similarity - verify content learning
                cosine_sim = sum([torch.nn.functional.cosine_similarity(emb, recon, dim=-1).mean() 
                                 for emb, recon in zip(anchor_embeddings, reconstructions)]) / len(anchor_embeddings)
                
                # Forward pass for positive (for contrastive loss)
                _, z_q_pos, _, _ = model(positive_embeddings, enable_progressive_masking=config.active_model_config.enable_progressive_masking)
                
                # Contrastive loss
                contrastive_loss = criterion(z_q, z_q_pos)
                
                # Total loss with adaptive weights
                loss = recon_weight * recon_loss + commitment_loss + contrastive_weight * contrastive_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log to TensorBoard
                writer.add_scalar('Loss/Total', loss.item(), global_step)
                writer.add_scalar('Loss/Reconstruction', recon_loss.item(), global_step)
                writer.add_scalar('Loss/Commitment', commitment_loss.item(), global_step)
                writer.add_scalar('Loss/Contrastive', contrastive_loss.item(), global_step)
                writer.add_scalar('Loss/Weighted_Reconstruction', (recon_weight * recon_loss).item(), global_step)
                writer.add_scalar('Loss/Weighted_Contrastive', (contrastive_weight * contrastive_loss).item(), global_step)
                writer.add_scalar('Weights/Reconstruction', recon_weight, global_step)
                writer.add_scalar('Weights/Contrastive', contrastive_weight, global_step)
                writer.add_scalar('CodeUsage/Total_Unique', total_unique_codes, global_step)
                writer.add_scalar('Metrics/Cosine_Similarity', cosine_sim.item(), global_step)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_commitment_loss += commitment_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_cosine_sim += cosine_sim.item()
                
                global_step += 1
                
                # Reset unused codes every 1000 batches (increased from 100 to avoid killing valid codes early in epoch)
                if config.active_model_config.enable_dead_code_revival and batch_idx > 0 and batch_idx % 1000 == 0:
                    with torch.no_grad():
                        for level in range(config.active_model_config.num_levels):
                            codebook_size = model.rqvae.codebook_sizes[level]
                            # Use epoch usage for revival to catch codes that die during training
                            # The > 1000 batch warmup ensures we don't kill codes just because they haven't been seen YET in this epoch
                            used_codes = epoch_unique_codes[level]
                            unused_codes = set(range(codebook_size)) - used_codes
                            
                            if len(unused_codes) > 0:
                                # Reset unused codes to random embeddings from current batch
                                # Get some random embeddings from the current batch
                                batch_embeddings = anchor_embeddings[0]  # (batch_size, 384)
                                
                                # For each unused code, assign a random embedding from the batch
                                revived_indices = []
                                for unused_idx in list(unused_codes)[:min(len(unused_codes), len(batch_embeddings))]:
                                    random_emb_idx = torch.randint(0, len(batch_embeddings), (1,)).item()
                                    # Encode the embedding to get the latent representation
                                    z = model.encoder([batch_embeddings[random_emb_idx:random_emb_idx+1]])
                                    # Assign to codebook
                                    model.rqvae.codebooks[level].weight[unused_idx] = z[0]
                                    revived_indices.append(unused_idx)
                                
                                # Mark revived codes as used in this epoch so they aren't immediately reset again
                                epoch_unique_codes[level].update(revived_indices)
                                all_time_unique_codes[level].update(revived_indices)
                                
                                if batch_idx % 500 == 0:  # Print less frequently
                                    print(f"  Reset {len(revived_indices)} unused codes at level {level}")

                
                # Print every 100 batches
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, "
                          f"Total: {loss.item():.4f}, Recon: {recon_loss.item():.4f} (w={recon_weight}), "
                          f"Commit: {commitment_loss.item():.4f}, Contrast: {contrastive_loss.item():.4f} (w={contrastive_weight}), "
                          f"CosSim: {cosine_sim.item():.4f}, Unique (Epoch): {epoch_total_unique}, Unique (Total): {total_unique_codes}")
                
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_commit = total_commitment_loss / len(dataloader)
            avg_contrast = total_contrastive_loss / len(dataloader)
            avg_cosine = total_cosine_sim / len(dataloader)
            
            # Log epoch averages
            writer.add_scalar('Epoch/Total_Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/Reconstruction_Loss', avg_recon, epoch)
            writer.add_scalar('Epoch/Commitment_Loss', avg_commit, epoch)
            writer.add_scalar('Epoch/Contrastive_Loss', avg_contrast, epoch)
            writer.add_scalar('Epoch/Cosine_Similarity', avg_cosine, epoch)
            
            # Log unique code usage per level
            for level in range(config.active_model_config.num_levels):
                unique_count = len(epoch_unique_codes[level])
                codebook_size = model.rqvae.codebook_sizes[level]
                usage_pct = (unique_count / codebook_size) * 100
                writer.add_scalar(f'CodeUsage/Level_{level}_Unique', unique_count, epoch)
                writer.add_scalar(f'CodeUsage/Level_{level}_Percentage', usage_pct, epoch)
                print(f"  Level {level}: {unique_count}/{codebook_size} codes used ({usage_pct:.1f}%)")
            
            print(f"Epoch {epoch+1}/{config.active_model_config.epochs}, Avg Loss: {avg_loss:.4f} "
                  f"(Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f}, Contrast: {avg_contrast:.4f}, CosSim: {avg_cosine:.4f})")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save model
        writer.close()
        os.makedirs(config.active_dataset.checkpoint_dir, exist_ok=True)
        save_path = os.path.join(config.active_dataset.checkpoint_dir, config.sid_model_checkpoint)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_sid()
