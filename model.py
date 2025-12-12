
# Minimal SmolLM2-135M style model implemented in PyTorch.
# Architecture: LLaMA-style decoder-only Transformer with:
# - RMSNorm
# - RoPE positional encoding
# - SwiGLU MLP
# - Grouped (GQA/MQA) attention: num_attention_heads != num_key_value_heads
#
# This file is self-contained (except PyTorch) and can be used as:
#
#   from model import SmolConfig, SmolLM2
#
#   cfg = SmolConfig.from_hf("HuggingFaceTB/SmolLM2-135M")
#   model = SmolLM2(cfg)

from dataclasses import dataclass
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1. Config

# Got config from HuggingFace Using:  transformers.AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Config: SmolLM2-135M

# LlamaConfig {
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "attention_bias": false,
#   "attention_dropout": 0.0,
#   "bos_token_id": 0,
#   "dtype": "bfloat16",
#   "eos_token_id": 0,
#   "head_dim": 64,
#   "hidden_act": "silu",
#   "hidden_size": 576,
#   "initializer_range": 0.041666666666666664,
#   "intermediate_size": 1536,
#   "is_llama_config": true,
#   "max_position_embeddings": 8192,
#   "mlp_bias": false,
#   "model_type": "llama",
#   "num_attention_heads": 9,
#   "num_hidden_layers": 30,
#   "num_key_value_heads": 3,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-05,
#   "rope_interleaved": false,
#   "rope_scaling": null,
#   "rope_theta": 100000,
#   "tie_word_embeddings": true,
#   "transformers_version": "4.57.3",
#   "use_cache": true,
#   "vocab_size": 49152
# }
# =========================

@dataclass
class SmolConfig:
    # Core dimensions
    vocab_size: int = 49152          # from HF config
    hidden_size: int = 576           # "hidden_size"
    intermediate_size: int = 1536    # "intermediate_size"
    num_hidden_layers: int = 30      # "num_hidden_layers"
    num_attention_heads: int = 9     # "num_attention_heads"
    num_key_value_heads: int = 3     # "num_key_value_heads"
    max_position_embeddings: int = 8192  # "max_position_embeddings"

    # Positional / RoPE
    rope_theta: float = 100000.0     # "rope_theta"

    # Norm / numerical
    rms_norm_eps: float = 1e-5       # "rms_norm_eps"

    # Biases
    attention_bias: bool = False     # "attention_bias"
    mlp_bias: bool = False           # "mlp_bias"

    # Misc
    dtype: torch.dtype = torch.bfloat16

    @property
    def head_dim(self) -> int:
        # Should be 64 for SmolLM2-135M (576 / 9).
        return self.hidden_size // self.num_attention_heads # 576 / 9 = 64

    @classmethod
    def from_hf(cls, hf_config) -> "SmolConfig":
        """
        Helper to build this config from a transformers LlamaConfig (Which is the config for the HuggingFace SmolLM2-135M model).
        Example:
            from transformers import AutoConfig
            hf = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
            cfg = SmolConfig.from_hf(hf)
        And then pass this config to this function call to set the config for the model.
        """
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads",
                                        hf_config.num_attention_heads),
            max_position_embeddings=hf_config.max_position_embeddings,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            rms_norm_eps=hf_config.rms_norm_eps,
            attention_bias=getattr(hf_config, "attention_bias", False),
            mlp_bias=getattr(hf_config, "mlp_bias", False),
            dtype=torch.bfloat16,  # SmolLM2 uses bfloat16
        )

# =========================
# 2. RMSNorm
# =========================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    Used in LLaMA / SmolLM2 instead of LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # rms = sqrt(mean(x^2)), but we can use rsqrt for stability
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

# =========================
# 3. RoPE (Rotary Positional Embeddings)
# =========================

def rope_freqs(head_dim: int, base: float, device, dtype):
    """
    Compute inverse frequencies for RoPE.
    """
    half_dim = head_dim // 2
    # Equivalent to: base^{ -2i / d }
    freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    return inv_freq  # shape: (half_dim,)

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cosine and sine caches for RoPE.
    Returns:
        cos: (1, 1, seq_len, head_dim/2)
        sin: (1, 1, seq_len, head_dim/2)
    """
    inv_freq = rope_freqs(head_dim, base, device, dtype)   # (half_dim,)
    # Positions
    t = torch.arange(seq_len, device=device, dtype=dtype)  # (seq_len,)
    freqs = torch.outer(t, inv_freq)                      # (seq_len, half_dim)
    cos = freqs.cos()[None, None, :, :]                   # (1,1,seq_len,half_dim)
    sin = freqs.sin()[None, None, :, :]                   # (1,1,seq_len,half_dim)
    return cos, sin

def apply_rope(
    x: torch.Tensor,  # (B, n_head, T, head_dim)
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to last dimension of x.
    cos, sin are broadcast to match (..., head_dim/2).
    """
    b, h, t, d = x.shape
    half = d // 2

    x1 = x[..., :half] # (B, n_head, T, head_dim/2)
    x2 = x[..., half:] # (B, n_head, T, head_dim/2)

    # cos/sin: (1,1,T,half) -> broadcast over B,h
    cos_t = cos[..., :t, :]
    sin_t = sin[..., :t, :]

    x1_rot = x1 * cos_t - x2 * sin_t
    x2_rot = x1 * sin_t + x2 * cos_t

    return torch.cat([x1_rot, x2_rot], dim=-1) # (B, n_head, T, head_dim)

# =========================
# 4. Attention
# =========================

class MultiHeadSelfAttention(nn.Module):
    """
    LLaMA / SmolLM2-style attention with:
    - Q heads = num_attention_heads
    - K/V heads = num_key_value_heads (GQA/MQA)
    - RoPE on Q and K
    - Causal masking
    """
    def __init__(self, config: SmolConfig):
        super().__init__()

        self.config = config
        self.n_heads = config.num_attention_heads # 9
        self.n_kv_heads = config.num_key_value_heads # 3
        self.head_dim = config.head_dim # 64
        self.hidden_size = config.hidden_size # 576

        assert self.hidden_size == self.n_heads * self.head_dim

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.n_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.o_proj.NANSmolLM_SCALE_INIT = True  # mark for scaled initialization

    def forward(
        self,
        x: torch.Tensor,                # (B, T, C) or (B, 1, C) for inference
        cos: torch.Tensor,              # (1,1,T,head_dim/2) or (1,1,1,head_dim/2) for inference
        sin: torch.Tensor,              # (1,1,T,head_dim/2) or (1,1,1,head_dim/2) for inference
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) or (B,1,1,T)
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (k_cache, v_cache)
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Projections: (B,T,C) -> (B,T,h,d) -> (B,h,T,d)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B,T,C) -> (B,T,h*d) -> (B,T,h,d) -> (B,h,T,d)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B,T,C) -> (B,T,k*d) -> (B,T,k,d) -> (B,k,T,d)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B,T,C) -> (B,T,v*d) -> (B,T,v,d) -> (B,v,T,d)

        # Apply RoPE to Q and K
        q = apply_rope(q, cos, sin)  # (B, h, T, d)
        k = apply_rope(k, cos, sin)  # (B, n_kv_heads, T, d)
        # v doesn't need RoPE

        # If using KV cache, concatenate with past keys/values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # past_k, past_v: (B, n_kv_heads, past_len, head_dim)
            k = torch.cat([past_k, k], dim=2)  # (B, n_kv_heads, past_len + T, head_dim)
            v = torch.cat([past_v, v], dim=2)  # (B, n_kv_heads, past_len + T, head_dim)
            seq_len = k.shape[2]
        else:
            seq_len = T

        # Store k, v for cache (before GQA expansion)
        k_cache = k  # (B, n_kv_heads, seq_len, head_dim)
        v_cache = v  # (B, n_kv_heads, seq_len, head_dim)

        # GQA: expand K/V if num_kv_heads < num_heads
        if self.n_kv_heads != self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)  # (B, n_kv_heads, seq_len, d) -> (B, n_heads, seq_len, d)
            v = v.repeat_interleave(repeat_factor, dim=1)  # (B, n_kv_heads, seq_len, d) -> (B, n_heads, seq_len, d)

        # Attention scores: (B,h,T,d) @ (B,h,d,seq_len) -> (B,h,T,seq_len)
        # Use Flash Attention instead of manual computation
        # Flash Attention handles causal masking internally with is_causal=True
        # For KV cache, we need to handle masking differently
        if past_key_value is None:
            # Full sequence: use causal mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            # With KV cache: no causal mask needed (already handled by cache structure)
            # But we still need to handle attention_mask if provided
            attn_mask = None
            if attention_mask is not None:
                # Convert attention_mask to the right format for Flash Attention
                if attention_mask.dim() == 2:
                    attn_mask = attention_mask[:, None, None, :]
                attn_mask = attn_mask.expand(B, self.n_heads, T, seq_len)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,  # Already handled by KV cache
                scale=1.0 / math.sqrt(self.head_dim),
            )

        # Reshape back: (B,T,C)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B,h,T,d) -> (B,T,h,d) -> (B,T,h*d) -> (B,T,C)
        out = self.o_proj(out) # (B,T,C) -> (B,T,C)
        
        # Return output and optionally the new KV cache
        present_key_value = None
        if use_cache:
            # Return k_cache, v_cache (before GQA expansion, after RoPE)
            present_key_value = (k_cache, v_cache)
        
        return out, present_key_value

# =========================
# 5. MLP (SwiGLU)
# =========================
class SmolMLP(nn.Module):
    """
    SwiGLU MLP:
        z = W1(x) -> split -> (x1, x2)
        out = W2( SiLU(x1) * x2 )
    """
    def __init__(self, config: SmolConfig):
        super().__init__()

        self.fc1 = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,   # for SwiGLU split (2 x 1536 = 3072)
            bias=config.mlp_bias,
        )

        self.fc2 = nn.Linear(
            config.intermediate_size,   # 1536
            config.hidden_size,   # 576
            bias=config.mlp_bias,
        )
        self.fc2.NANSmolLM_SCALE_INIT = True     # mark for scaled initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)# (B,T,C) -> (B,T,2*intermediate_size) -> (B,T,1536*2) -> (B,T,3072)
        x1, x2 = x.chunk(2, dim=-1)  # (B,T,2*intermediate_size) = (B,T,3072) -> (B,T,intermediate), (B,T,intermediate) = (B,T,1536), (B,T,1536)
        return self.fc2(F.silu(x1) * x2) # (B,T,intermediate) * (B,T,intermediate) -> (B,T,intermediate) -> (B,T,hidden_size) = (B,T,576)


# =========================
# 6. Transformer Block
# =========================
class SmolBlock(nn.Module):
    def __init__(self, config: SmolConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SmolMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + residual for attention
        attn_out, present_key_value = self.attn(
            self.attn_norm(x), cos, sin, attention_mask, past_key_value, use_cache
        )
        x = x + attn_out
        # Pre-norm + residual for MLP
        x = x + self.mlp(self.mlp_norm(x))
        return x, present_key_value

# =============================================
# 7. Top-level SmolLM2-135M Model Architecture
#  SmolLM2 follows the LLaMA-style decoder-only Transformer architecture.
# =============================================
class SmolLM2(nn.Module):
    """
    SmolLM2-135M-style LLaMA decoder-only language model.

    Usage:
        cfg = SmolConfig()
        model = SmolLM2(cfg)

        input_ids: LongTensor (B, T)
        logits = model(input_ids)
    """
    def __init__(self, config: SmolConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        ) # (Vocab_Size, Hidden_Size) (49152 x 576)

        self.layers = nn.ModuleList(
            [SmolBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        ) # (Hidden_Size, Vocab_Size) (576 x 49152)

        # tie weights
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for Linear and Embedding layers.
        For Linear layers with NANGPT_SCALE_INIT attribute, scale std by sqrt(2 * num_layers).
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANSmolLM_SCALE_INIT'):
                # Initialize marked linear layers using formula: std = 0.02 * sqrt(2 * num_layers) this 2 x number of layers is because of each block has 2 residual connection. So it actually based on number of residual connection in the model.
                std *= (2 * self.config.num_hidden_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 1 / math.sqrt(module.weight.shape[1])) # std should be calculated using embedding vector size with formula: std = 1 / sqrt(embedding_vector_size)

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.shape
        
        # For inference with KV cache, we might have T=1
        if past_key_values is None:
            assert T <= self.config.max_position_embeddings, (
                f"Sequence length {T} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
            seq_len = T
        else:
            # With KV cache, current sequence length is past_len + T
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            seq_len = past_len + T
            assert seq_len <= self.config.max_position_embeddings, (
                f"Total sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        # Embedding
        x = self.embed_tokens(input_ids)  # (B,T) -> (B,T,C)

        # RoPE cache - build for the full sequence length (past + current)
        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.config.head_dim,
            base=self.config.rope_theta,
            device=x.device,
            dtype=x.dtype,
        )
        
        # If using KV cache, we only need cos/sin for current positions
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            # Slice to get only the current positions for RoPE
            cos = cos[..., past_len:, :]
            sin = sin[..., past_len:, :]

        # Layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, cos, sin, attention_mask, past_kv, use_cache)
            if use_cache:
                present_key_values.append(present_kv)

        # Final norm + lm head
        x = self.norm(x)
        logits = self.lm_head(x)  # (B,T,C) -> (B,T,vocab_size)
        return logits, present_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using KV cache for efficient inference.
        
        Args:
            input_ids: (B, T) input token ids
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (keep top k tokens)
            top_p: nucleus sampling (keep tokens with cumulative probability <= top_p)
            eos_token_id: end-of-sequence token id (stop generation when encountered)
        
        Returns:
            generated_ids: (B, T + max_new_tokens) generated token ids
        """
        self.eval()
        device = input_ids.device
        B, T = input_ids.shape
        
        # Start with input_ids
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Forward pass with KV cache
            # On first iteration, use full input_ids. On subsequent iterations, use only last token
            if past_key_values is None:
                # First iteration: process full sequence
                current_input = generated_ids
            else:
                # Subsequent iterations: only process the last generated token
                current_input = generated_ids[:, -1:]
            
            logits, past_key_values = self.forward(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # Get logits for the last token (always the last position in logits)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated_ids

# =========================
# 8. Quick self-test
# =========================
if __name__ == "__main__":
    # Tiny sanity check: runs a forward pass on random input
    cfg = SmolConfig()
    model = SmolLM2(cfg)

    B, T = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, _ = model(x)

    print("Input shape :", x.shape)
    print("Logits shape:", logits.shape)  # should be (2, 16, vocab_size)
