import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class PLUM_LLM(nn.Module):
    """
    PLUM Generative Retrieval Model using pre-trained GPT-2.
    Extends the vocabulary with Semantic ID tokens.
    """
    def __init__(self, model_name: str = 'gpt2', num_levels: int = 3, base_codebook_size: int = 512):
        super().__init__()
        
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add pad token if missing (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
        # Calculate SID tokens
        self.num_levels = num_levels
        self.codebook_sizes = [int(base_codebook_size / (2**l)) for l in range(num_levels)]
        self.total_sid_tokens = sum(self.codebook_sizes)
        
        # Add SID tokens to tokenizer
        # Format: <SID_L{level}_{code}>
        new_tokens = []
        self.sid_token_map = {} # Map (level, code) -> token_string
        
        for l in range(num_levels):
            for c in range(self.codebook_sizes[l]):
                token = f"<SID_L{l}_{c}>"
                new_tokens.append(token)
                self.sid_token_map[(l, c)] = token
                
        # Add tokens to tokenizer
        num_added_toks = self.tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added_toks} SID tokens to vocabulary.")
        
        # Resize model embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_sid_token_id(self, level: int, code: int) -> int:
        token_str = self.sid_token_map.get((level, code))
        if token_str:
            return self.tokenizer.convert_tokens_to_ids(token_str)
        return None
        
    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

# Helper to get tokenizer separately if needed, but usually attached to model wrapper
def get_plum_tokenizer(model_name='gpt2'):
    return GPT2Tokenizer.from_pretrained(model_name)
