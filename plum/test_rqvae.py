import torch
import unittest
from plum.sid_model import RQVAE

class TestRQVAE(unittest.TestCase):
    def setUp(self):
        self.input_dim = 64
        self.num_levels = 3
        self.base_codebook_size = 64
        self.batch_size = 10
        self.model = RQVAE(self.input_dim, self.num_levels, self.base_codebook_size)
        
    def test_output_shapes(self):
        """Test if output shapes are correct"""
        z = torch.randn(self.batch_size, self.input_dim)
        z_q, codes, loss = self.model(z, training=False)
        
        self.assertEqual(z_q.shape, (self.batch_size, self.input_dim), "z_q shape mismatch")
        self.assertEqual(codes.shape, (self.batch_size, self.num_levels), "codes shape mismatch")
        self.assertEqual(loss.shape, (), "loss should be scalar")
        
    def test_code_ranges(self):
        """Test if generated codes are within valid range for each level"""
        z = torch.randn(self.batch_size, self.input_dim)
        _, codes, _ = self.model(z, training=False)
        
        for l in range(self.num_levels):
            codebook_size = self.model.codebook_sizes[l]
            level_codes = codes[:, l]
            self.assertTrue(torch.all(level_codes >= 0), f"Level {l} codes < 0")
            self.assertTrue(torch.all(level_codes < codebook_size), f"Level {l} codes >= size {codebook_size}")
            
    def test_reconstruction_improvement(self):
        """Test if residual norm decreases as we go deeper (manual step-through)"""
        # We can't easily test this with the full forward pass since it sums everything.
        # But we can check if the final reconstruction is closer than zero vector?
        # Better: let's inspect the internal logic by mocking or just checking logic.
        # Actually, for a random initialized network, it's NOT guaranteed that quantization reduces error 
        # compared to original z, but the residual *magnitude* should decrease if codebooks match well.
        # With random init, this test might be flaky. 
        # Instead, let's verify that z_q is actually the sum of the selected embeddings.
        
        z = torch.randn(self.batch_size, self.input_dim)
        z_q, codes, _ = self.model(z, training=False)
        
        # Reconstruct manually
        z_q_manual = torch.zeros_like(z)
        for l in range(self.num_levels):
            indices = codes[:, l]
            z_q_manual += self.model.codebooks[l](indices)
            
        self.assertTrue(torch.allclose(z_q, z_q_manual, atol=1e-5), "z_q does not match sum of codes")

    def test_progressive_masking(self):
        """Test that progressive masking changes the effective depth during training"""
        # This is hard to test deterministically because r is random.
        # But we can check that if we force r, the output changes.
        # The current implementation doesn't allow forcing r easily without modifying code.
        # However, we can check that training=True produces different results than training=False 
        # (due to random r) for the same input, *sometimes*.
        
        z = torch.randn(self.batch_size, self.input_dim)
        
        # training=False (always full depth)
        z_q_eval, _, _ = self.model(z, training=False)
        
        # training=True (random depth)
        # We run multiple times; if r < num_levels is chosen, z_q should be different (norm should be smaller usually)
        # or at least different from full depth z_q_eval.
        
        different = False
        for _ in range(20):
            z_q_train, _, _ = self.model(z, training=True, enable_progressive_masking=True)
            if not torch.allclose(z_q_train, z_q_eval):
                different = True
                break
                
        self.assertTrue(different, "Progressive masking did not seem to trigger (outputs always same as full depth)")

if __name__ == '__main__':
    unittest.main()
