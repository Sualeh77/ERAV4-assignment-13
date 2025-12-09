# SmolLM2-135M Training Guide

This directory contains the training code for SmolLM2-135M model.

## Files

- `model.py`: Model definition with KV cache support for inference
- `train.py`: Main training script (trains for 5000 steps)
- Run with checkpoint path to Resume training for 50 additional steps

## Setup

Install required packages:

```bash
pip install torch lightning transformers tensorboard
```

## Training

### Phase 1: Initial Training (5000 steps)

Run the main training script:

```bash
python train.py
```

This will:
- Train the model for 5000 steps
- Generate text predictions every 500 steps
- Save checkpoints every 500 steps
- Log training metrics to TensorBoard and text file
- Save the final checkpoint at step 5000

### Phase 2: Resume Training (50 additional steps)

After Phase 1 completes, run:

```bash
python train.py
```

But this time set the checkpoint path, and set steps as 50 to resume training for 50 additional steps. just to showcase that training is started where it stopped.


This will:
- Load the checkpoint from Phase 1
- Train for 50 additional steps
- Save the final checkpoint

## Training Configuration

The training uses the following hyperparameters (from the SmolLM2 paper):

- **Optimizer**: AdamW with (β₁, β₂) = (0.9, 0.95)
- **Learning Rate Schedule**: Warmup Stable Decay (WSD)
  - Warmup: 2000 steps
  - Peak LR: 5.0 × 10⁻⁴
  - Stable phase: maintains peak LR
  - Decay: reduces to zero over 10% of total steps
- **Block size**: 512 tokens
- **Batch size**: 4
- **Precision**: bfloat16 (if GPU available), float32 otherwise

## Outputs

- **Checkpoints**: Saved in `./checkpoints/`
- **TensorBoard logs**: Saved in `./logs/tensorboard/`
- **Text logs**: Saved in `./logs/training_*.log`

## Model Features

The model includes:
- **KV Cache**: Efficient inference using key-value caching
- **Generation**: Text generation with top-k and top-p sampling
- **Checkpointing**: Full state saving for resuming training

## Usage Example

```python
from model import SmolLM2, SmolConfig
from transformers import AutoTokenizer, AutoConfig

# Load config
hf_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
config = SmolConfig.from_hf(hf_config)

# Create model
model = SmolLM2(config)

# Load checkpoint
checkpoint = torch.load("checkpoints/smollm2-00500-*.ckpt")
model.load_state_dict(checkpoint['state_dict'])

# Generate text
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
prompt = "First Citizen:"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

generated_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
)

generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)
```
