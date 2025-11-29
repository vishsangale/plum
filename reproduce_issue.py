import torch
import torch.nn as nn
from plum.sid_model import RQVAE

def test_progressive_masking_gradients():
    # Setup
    input_dim = 32
    num_levels = 4
    batch_size = 2
    
    model = RQVAE(input_dim=input_dim, num_levels=num_levels)
    
    # Mock input
    z = torch.randn(batch_size, input_dim)
    
    # We want to force r=1 (only first level used)
    # We can patch torch.randint or just modify the model temporarily if possible.
    # But patching is cleaner for a reproduction script.
    
    # Let's use a monkey patch on torch.randint to always return 1
    original_randint = torch.randint
    
    def mocked_randint(low, high, size, **kwargs):
        return torch.tensor([1])
        
    torch.randint = mocked_randint
    
    try:
        # Forward pass
        z_q, codes, loss = model(z, training=True, enable_progressive_masking=True)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        print(f"Loss: {loss.item()}")
        
        for l in range(num_levels):
            grad = model.codebooks[l].weight.grad
            if grad is not None:
                grad_norm = grad.norm().item()
                print(f"Level {l+1} gradient norm: {grad_norm}")
                
                if l == 0:
                    if grad_norm == 0:
                        print("ERROR: Level 1 should have gradients!")
                else:
                    if grad_norm > 0:
                        print(f"ISSUE: Level {l+1} has gradients but r=1! This confirms the bug.")
                    else:
                        print(f"Level {l+1} has no gradients (Correct behavior).")
            else:
                print(f"Level {l+1} has no gradient (None).")
                
    finally:
        torch.randint = original_randint

if __name__ == "__main__":
    test_progressive_masking_gradients()
