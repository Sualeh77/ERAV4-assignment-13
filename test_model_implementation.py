import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model import SmolLM2, SmolConfig  # your implementation


PRETRAINED_NAME = "HuggingFaceTB/SmolLM2-135M"


def build_custom_model():
    """Create our SmolLM2 using HF config to ensure identical hyperparams."""
    hf_cfg = AutoConfig.from_pretrained(PRETRAINED_NAME)
    cfg = SmolConfig.from_hf(hf_cfg)
    model = SmolLM2(cfg)
    return model, cfg


def build_hf_model():
    """Load reference HF model."""
    hf_model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_NAME,
        torch_dtype=torch.float32,  # use float32 for easier comparison
    )
    hf_model.eval()
    return hf_model


def load_weights_from_hf(custom_model: SmolLM2, hf_model: AutoModelForCausalLM):
    """
    Map HF LlamaForCausalLM weights into our SmolLM2 model.

    - HF model structure: hf_model.model (LlamaModel) + hf_model.lm_head
    - Our model: embed_tokens, layers, norm, lm_head
    """
    hf_state = hf_model.state_dict()
    custom_state = custom_model.state_dict()

    # 1. Embeddings
    custom_state["embed_tokens.weight"] = hf_state["model.embed_tokens.weight"]

    # 2. Per-layer mappings
    num_layers = custom_model.config.num_hidden_layers

    for i in range(num_layers):
        # Norms
        custom_state[f"layers.{i}.attn_norm.weight"] = hf_state[
            f"model.layers.{i}.input_layernorm.weight"
        ]
        custom_state[f"layers.{i}.mlp_norm.weight"] = hf_state[
            f"model.layers.{i}.post_attention_layernorm.weight"
        ]

        # Attention projections
        custom_state[f"layers.{i}.attn.q_proj.weight"] = hf_state[
            f"model.layers.{i}.self_attn.q_proj.weight"
        ]
        custom_state[f"layers.{i}.attn.k_proj.weight"] = hf_state[
            f"model.layers.{i}.self_attn.k_proj.weight"
        ]
        custom_state[f"layers.{i}.attn.v_proj.weight"] = hf_state[
            f"model.layers.{i}.self_attn.v_proj.weight"
        ]
        custom_state[f"layers.{i}.attn.o_proj.weight"] = hf_state[
            f"model.layers.{i}.self_attn.o_proj.weight"
        ]

        # MLP: HF has gate_proj, up_proj, down_proj
        gate = hf_state[f"model.layers.{i}.mlp.gate_proj.weight"]
        up = hf_state[f"model.layers.{i}.mlp.up_proj.weight"]
        down = hf_state[f"model.layers.{i}.mlp.down_proj.weight"]

        # Our fc1 is [gate; up] concatenated along output dim (dim=0)
        custom_state[f"layers.{i}.mlp.fc1.weight"] = torch.cat([gate, up], dim=0)
        # Our fc2 is down_proj
        custom_state[f"layers.{i}.mlp.fc2.weight"] = down

    # 3. Final norm
    custom_state["norm.weight"] = hf_state["model.norm.weight"]

    # 4. LM head (tied with embeddings, but we still load it)
    custom_state["lm_head.weight"] = hf_state["lm_head.weight"]

    # Now load into the model
    missing, unexpected = custom_model.load_state_dict(custom_state, strict=False)
    return missing, unexpected


def test_weight_loading():
    """
    1. Build custom SmolLM2 model (our implementation).
    2. Build HF reference model.
    3. Load HF weights into our model via mapping.
    4. Run a small test prompt and compare logits.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("🟦 Building custom model...")
    custom_model, cfg = build_custom_model()
    custom_model.to(device)
    custom_model.eval()

    print("🟦 Building HF reference model...")
    hf_model = build_hf_model()
    hf_model.to(device)

    print("🟦 Mapping HF weights into custom model...")
    missing, unexpected = load_weights_from_hf(custom_model, hf_model)

    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")
    if missing:
        print("  Missing examples:", missing[:5])
    if unexpected:
        print("  Unexpected examples:", unexpected[:5])

    if len(missing) > 0:
        print("⚠️ There are missing keys; mapping may be incomplete.")
    else:
        print("✅ All expected parameters were assigned from HF weights.")

    # 5. Test with a dummy input
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("🟦 Running HF model forward...")
    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits  # (B, T, V)

    print("🟦 Running custom model forward...")
    with torch.no_grad():
        custom_logits, _ = custom_model(inputs["input_ids"])

    # 6. Compare logits
    # align dtypes
    hf_logits = hf_logits.to(torch.float32)
    custom_logits = custom_logits.to(torch.float32)

    diff = torch.abs(hf_logits - custom_logits).max().item()
    print(f"🔍 Max absolute difference between logits: {diff:.6f}")

    if diff < 1e-4:
        print("✅ SUCCESS: Outputs match very closely. Implementation is correct.")
    elif diff < 1e-2:
        print("🟡 Outputs are close but not identical; check for small implementation differences (e.g., RoPE details).")
    else:
        print("❌ Outputs differ significantly. Some part of the implementation is likely off.")

    # 7. Print predictions from both models
    print("\n📝 Predictions:")
    print(f"Prompt: '{prompt}'")
    
    # Get predicted token IDs (argmax on vocabulary dimension)
    hf_predicted_ids = hf_logits.argmax(dim=-1)  # (B, T)
    custom_predicted_ids = custom_logits.argmax(dim=-1)  # (B, T)
    
    # Get the next token prediction (last position)
    hf_next_token_id = hf_predicted_ids[0, -1].item()
    custom_next_token_id = custom_predicted_ids[0, -1].item()
    
    # Decode the next token
    hf_next_token = tokenizer.decode([hf_next_token_id])
    custom_next_token = tokenizer.decode([custom_next_token_id])
    
    print(f"HF Model prediction (next token): '{hf_next_token}' (token_id: {hf_next_token_id})")
    print(f"Custom Model prediction (next token): '{custom_next_token}' (token_id: {custom_next_token_id})")
    
    # Also show full sequence predictions for comparison
    hf_full_prediction = tokenizer.decode(hf_predicted_ids[0])
    custom_full_prediction = tokenizer.decode(custom_predicted_ids[0])
    print(f"\nHF Model full sequence prediction: '{hf_full_prediction}'")
    print(f"Custom Model full sequence prediction: '{custom_full_prediction}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model_implementation.py test_weight_loading")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "test_weight_loading":
        test_weight_loading()
    else:
        print(f"Unknown mode: {mode}")
