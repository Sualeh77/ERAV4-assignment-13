# SmolLM2-135M Implementation

A from-scratch PyTorch implementation of the SmolLM2-135M language model, following the LLaMA architecture with modern optimizations.

## Overview

This repository contains a complete implementation of SmolLM2-135M, a 135 million parameter decoder-only transformer model. The implementation includes:

- **Model Architecture** (`model.py`): Complete model definition with KV cache support
- **Training Script** (`train.py`): PyTorch Lightning training with WSD scheduler
- **Gradio App** (`app.py`): Interactive web interface for text generation

## Model Architecture (`model.py`)

### Architecture Components

The model follows the LLaMA-style decoder-only transformer architecture with the following key components:

#### 1. **SmolConfig** (Configuration Class)

A dataclass that stores all model hyperparameters:

```python
@dataclass
class SmolConfig:
    vocab_size: int = 49152          # Vocabulary size
    hidden_size: int = 576           # Hidden dimension
    intermediate_size: int = 1536     # MLP intermediate dimension
    num_hidden_layers: int = 30      # Number of transformer layers
    num_attention_heads: int = 9      # Number of query heads
    num_key_value_heads: int = 3     # Number of key/value heads (GQA)
    max_position_embeddings: int = 8192  # Maximum sequence length
    rope_theta: float = 100000.0     # RoPE base frequency
    rms_norm_eps: float = 1e-5       # RMSNorm epsilon
    attention_bias: bool = False     # Whether to use bias in attention
    mlp_bias: bool = False           # Whether to use bias in MLP
    dtype: torch.dtype = torch.bfloat16
```

**Key Features:**
- `head_dim` property: Automatically computes head dimension (hidden_size // num_attention_heads = 64)
- `from_hf()` class method: Loads configuration from HuggingFace model config

#### 2. **RMSNorm** (Root Mean Square Normalization)

Replaces LayerNorm with a more efficient normalization:

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x
```

**Benefits:**
- More efficient than LayerNorm (no mean subtraction)
- Used throughout the model for pre-norm architecture

#### 3. **RoPE** (Rotary Positional Embeddings)

Rotary Position Embeddings applied to query and key tensors:

```python
def build_rope_cache(seq_len, head_dim, base, device, dtype):
    # Computes cosine and sine caches for RoPE
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return cos, sin

def apply_rope(x, cos, sin):
    # Applies rotary transformation to input tensor
    x1, x2 = x[..., :half], x[..., half:]
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    return torch.cat([x1_rot, x2_rot], dim=-1)
```

**Key Features:**
- Relative positional encoding (no absolute position embeddings)
- Applied only to Q and K (not V)
- Supports efficient caching for inference

#### 4. **MultiHeadSelfAttention** (Grouped Query Attention)

Implements GQA (Grouped Query Attention) where:
- **Query heads**: 9 (full attention)
- **Key/Value heads**: 3 (shared across query heads)

```python
class MultiHeadSelfAttention(nn.Module):
    def forward(self, x, cos, sin, past_key_value=None, use_cache=False):
        # 1. Project to Q, K, V
        q = self.q_proj(x)  # (B, T, n_heads * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)
        
        # 2. Apply RoPE to Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # 3. KV Cache support (for inference)
        if past_key_value:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # 4. GQA: Expand K/V if needed
        if n_kv_heads < n_heads:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # 5. Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        scores = scores + causal_mask  # Causal masking
        
        # 6. Softmax and weighted sum
        probs = F.softmax(scores, dim=-1)
        out = probs @ v
        
        return out, present_key_value
```

**Key Features:**
- **KV Cache**: Efficient inference by caching past key-value pairs
- **GQA**: Reduces memory by sharing K/V heads (3:1 ratio)
- **Causal Masking**: Prevents attending to future tokens
- **RoPE Integration**: Positional encoding via rotary embeddings

#### 5. **SmolMLP** (SwiGLU Activation)

Implements the SwiGLU (Swish-Gated Linear Unit) MLP:

```python
class SmolMLP(nn.Module):
    def forward(self, x):
        # fc1 outputs 2 * intermediate_size
        x = self.fc1(x)  # (B, T, 2 * 1536) = (B, T, 3072)
        x1, x2 = x.chunk(2, dim=-1)  # Split into two parts
        # SwiGLU: SiLU(x1) * x2
        return self.fc2(F.silu(x1) * x2)
```

**Key Features:**
- **SwiGLU**: `SiLU(x1) * x2` activation (better than ReLU/GELU)
- **No bias**: Following LLaMA architecture
- **Efficient**: Single matrix multiplication with split

#### 6. **SmolBlock** (Transformer Block)

Combines attention and MLP with pre-norm and residual connections:

```python
class SmolBlock(nn.Module):
    def forward(self, x, cos, sin, past_key_value=None, use_cache=False):
        # Pre-norm attention with residual
        attn_out, present_kv = self.attn(
            self.attn_norm(x), cos, sin, 
            past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + attn_out
        
        # Pre-norm MLP with residual
        x = x + self.mlp(self.mlp_norm(x))
        
        return x, present_kv
```

**Architecture:**
- **Pre-norm**: Normalization before attention/MLP (not after)
- **Residual connections**: Skip connections for gradient flow
- **KV Cache passthrough**: Supports efficient inference

#### 7. **SmolLM2** (Main Model)

Top-level model that combines all components:

```python
class SmolLM2(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([SmolBlock(config) for _ in range(30)])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Weight tying: share embeddings and output weights
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(self, input_ids, past_key_values=None, use_cache=False):
        # 1. Token embeddings
        x = self.embed_tokens(input_ids)
        
        # 2. Build RoPE cache
        cos, sin = build_rope_cache(...)
        
        # 3. Pass through transformer layers
        present_key_values = []
        for layer in self.layers:
            x, present_kv = layer(x, cos, sin, past_key_value, use_cache)
            if use_cache:
                present_key_values.append(present_kv)
        
        # 4. Final norm and language modeling head
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, present_key_values
```

**Key Features:**
- **Weight Tying**: Embeddings and output weights are shared (reduces parameters)
- **KV Cache Support**: Full support for efficient autoregressive generation
- **30 Layers**: Deep transformer stack for capacity

#### 8. **Generate Method** (Text Generation)

Autoregressive text generation with KV cache:

```python
@torch.no_grad()
def generate(self, input_ids, max_new_tokens=100, temperature=1.0, 
             top_k=None, top_p=None, eos_token_id=None):
    generated = input_ids
    past_key_values = None
    
    for _ in range(max_new_tokens):
        # Forward pass with KV cache
        logits, past_key_values = self.forward(
            generated[:, -1:] if past_key_values else generated,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        # Sample next token with temperature, top-k, top-p
        next_token_logits = logits[:, -1, :] / temperature
        # Apply top-k and top-p filtering
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        if eos_token_id and (next_token == eos_token_id).all():
            break
    
    return generated
```

**Key Features:**
- **KV Cache**: Only processes new tokens (not entire sequence)
- **Sampling**: Supports temperature, top-k, and top-p (nucleus) sampling
- **Efficient**: O(1) per token after initial forward pass

### Model Specifications

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | ~135M |
| **Hidden Size** | 576 |
| **Layers** | 30 |
| **Attention Heads** | 9 (Q), 3 (K/V) |
| **Head Dimension** | 64 |
| **Intermediate Size** | 1536 |
| **Vocabulary Size** | 49,152 |
| **Max Sequence Length** | 8,192 |
| **RoPE Theta** | 100,000 |
| **Activation** | SwiGLU (SiLU-gated) |
| **Normalization** | RMSNorm |
| **Weight Tying** | Yes (embeddings = output) |

### Key Design Choices

1. **GQA (Grouped Query Attention)**: 3:1 ratio reduces memory by 66% for K/V cache
2. **Pre-norm Architecture**: More stable training than post-norm
3. **RMSNorm**: Faster and simpler than LayerNorm
4. **RoPE**: Relative positional encoding, no learned embeddings
5. **SwiGLU**: Better activation than ReLU/GELU
6. **Weight Tying**: Reduces parameters and improves generalization
7. **No Biases**: Following LLaMA, reduces parameters slightly

### Usage Example

```python
from model import SmolConfig, SmolLM2
from transformers import AutoConfig

# Load config from HuggingFace
hf_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
config = SmolConfig.from_hf(hf_config)

# Create model
model = SmolLM2(config)

# Forward pass (training)
input_ids = torch.randint(0, config.vocab_size, (2, 512))
logits, _ = model(input_ids, use_cache=False)

# Text generation (inference with KV cache)
prompt_ids = tokenizer.encode("Hello, how are you?")
generated = model.generate(
    prompt_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)
```

## Training

See `README_TRAINING.md` for detailed training instructions.

## Inference

See `app.py` for the Gradio web interface or use the `generate()` method directly.

## References

- [SmolLM2 Paper](https://arxiv.org/abs/2406.02528)
- [LLaMA Architecture](https://arxiv.org/abs/2302.13971)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [SwiGLU Activation](https://arxiv.org/abs/2002.05202)
