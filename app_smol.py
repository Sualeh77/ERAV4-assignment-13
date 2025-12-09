"""
Gradio app for SmolLM2-135M inference with streaming output.
Uses Lightning checkpoint saved from training.
"""

import sys
from pathlib import Path
from typing import List, Optional

import gradio as gr
import torch
from transformers import AutoConfig, AutoTokenizer

from model import SmolConfig, SmolLM2
from train import SmolLM2Module

# Device setup
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"

# Globals
model: Optional[SmolLM2] = None
tokenizer = None

# Allow SmolConfig to be deserialized from Lightning checkpoints when torch.load
try:
    torch.serialization.add_safe_globals([SmolConfig])  # type: ignore[attr-defined]
except Exception:
    pass


def load_model_checkpoint(checkpoint_path: str = "checkpoints/smollm2-final-step-05000.ckpt"):
    """Load Lightning checkpoint and return status string."""
    global model, tokenizer

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return f"❌ Checkpoint not found: {ckpt}"

    try:
        hf_cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        config = SmolConfig.from_hf(hf_cfg)
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        module = SmolLM2Module.load_from_checkpoint(
            str(ckpt),
            config=config,
            tokenizer=tokenizer,
            map_location=DEVICE,
            strict=False,
        )
        module.eval()
        model = module.model.to(DEVICE).eval()
        return f"✅ Model loaded from {ckpt} on {DEVICE}"
    except Exception as e:  # pragma: no cover - interactive
        model = None
        return f"❌ Error loading model: {e}"


def stream_generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    """Generator that yields only the generated text (without prompt)."""
    global model, tokenizer
    if model is None or tokenizer is None:
        yield "⚠️ Load the model first (click Reload Model)."
        return

    if not prompt or not prompt.strip():
        yield "⚠️ Please enter a prompt."
        return

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(DEVICE)

    # Guard against context overflow
    if input_ids.shape[1] >= model.config.max_position_embeddings:
        yield f"⚠️ Prompt too long ({input_ids.shape[1]} tokens). Max is {model.config.max_position_embeddings}."
        return

    generated = input_ids
    past_key_values: Optional[List] = None
    prompt_length = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                current_input = generated
            else:
                current_input = generated[:, -1:]

            logits, past_key_values = model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)

            # top-k
            if top_k > 0:
                values, _ = torch.topk(next_token_logits, top_k)
                min_keep = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_keep,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                sorted_mask = cumulative > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_token_logits = torch.where(mask, torch.full_like(next_token_logits, float("-inf")), next_token_logits)

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            # Decode only the generated part (skip the prompt)
            generated_text = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
            yield generated_text


# Initial load
INITIAL_STATUS = load_model_checkpoint()


def chat_stream(message, history, max_tokens, temperature, top_k, top_p):
    """Gradio wrapper for streaming chat."""
    if history is None:
        history = []

    # Convert history from tuple format to dict format if needed
    if history and isinstance(history[0], (list, tuple)):
        # Convert from tuple format [(user, assistant), ...] to dict format
        new_history = []
        for h in history:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                if h[0]:  # User message
                    new_history.append({"role": "user", "content": str(h[0])})
                if h[1]:  # Assistant message
                    new_history.append({"role": "assistant", "content": str(h[1])})
        history = new_history

    # Append user message
    user_msg = (message or "").strip()
    if not user_msg:
        yield history
        return
    
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ""})

    stream = stream_generate(user_msg, max_tokens, temperature, top_k, top_p)
    for partial in stream:
        # Update the last assistant message with generated text
        if partial:
            history[-1] = {"role": "assistant", "content": str(partial)}
        yield history


def clear_chat():
    return "", []


with gr.Blocks(title="SmolLM2-135M Text Generator") as demo:
    gr.Markdown(
        """
        # 🤖 SmolLM2-135M Text Generator

        Generate text with your trained SmolLM2-135M checkpoint (streaming output).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Status")
            status_text = gr.Textbox(value=INITIAL_STATUS, label="Status", interactive=False, lines=2)
            load_btn = gr.Button("🔄 Reload Model", variant="secondary")
            ckpt_input = gr.Textbox(
                value="checkpoints/smollm2-step=05000-train_loss=0.0918.ckpt",
                label="Checkpoint path",
                interactive=True,
            )
            load_btn.click(fn=lambda p: load_model_checkpoint(p), inputs=ckpt_input, outputs=status_text)

            gr.Markdown("### Generation Parameters")
            max_tokens = gr.Slider(10, 500, value=100, step=10, label="Max Tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=50, step=5, label="Top-K")
            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-P")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat Interface")
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                msg = gr.Textbox(label="Your Message", placeholder="Type your prompt here...", scale=4, lines=2)
                submit_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")

    msg.submit(fn=chat_stream, inputs=[msg, chatbot, max_tokens, temperature, top_k, top_p], outputs=chatbot)
    submit_btn.click(fn=chat_stream, inputs=[msg, chatbot, max_tokens, temperature, top_k, top_p], outputs=chatbot).then(fn=lambda: "", outputs=msg)
    clear_btn.click(fn=clear_chat, outputs=[msg, chatbot])


if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
