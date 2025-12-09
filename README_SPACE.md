# SmolLM2-135M Hugging Face Space Setup Guide

This guide explains how to push your model and app to Hugging Face Spaces.

## Files Needed for Hugging Face Space

1. **app.py** - Main Gradio application (already created)
2. **model.py** - Model definition
3. **train.py** - Contains SmolLM2Module class (needed for loading checkpoints)
4. **requirements.txt** - Python dependencies
5. **README.md** - Space description (optional but recommended)

## Step-by-Step Guide

### 1. Fix Merge Conflicts (if still present)

If you still have merge conflicts, resolve them:
```bash
# Check status
git status

# Resolve conflicts in train.py and pyproject.toml
# Then commit
git add train.py pyproject.toml
git commit -m "Resolve merge conflicts"
```

### 2. Create Hugging Face Space (if not already created)

```bash
# Create the space (without --sdk flag, set it in web UI)
huggingface-cli repo create smollm2-135m-trained-on-tinyShakespear-forfun --type=space
```

Then go to the Space settings in the web UI and set:
- **SDK**: Gradio
- **Python version**: 3.12

### 3. Add Hugging Face Remote

```bash
# Add HF Space as remote (different name to avoid confusion with GitHub)
git remote add huggingface https://huggingface.co/spaces/Sualeh77/smollm2-135m-trained-on-tinyShakespear-forfun
```

### 4. Prepare Files for Space

Make sure these files are ready:
- ✅ `app.py` - Main app (loads from HF model repo)
- ✅ `model.py` - Model definition
- ✅ `train.py` - Contains SmolLM2Module
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Should exclude logs/, checkpoints/, etc.

### 5. Push to Hugging Face Space

```bash
# First, disable GPG signing temporarily (if you had issues)
git config --global commit.gpgsign false

# Add and commit files
git add app.py model.py train.py requirements.txt .gitignore
git commit -m "Add Gradio app for SmolLM2-135M inference"

# Push to Hugging Face Space
git push huggingface main

# Re-enable GPG signing if you want
git config --global commit.gpgsign true
```

### 6. Verify on Hugging Face

1. Go to your Space: https://huggingface.co/spaces/Sualeh77/smollm2-135m-trained-on-tinyShakespear-forfun
2. Check the "Files" tab - you should see `app.py`, `model.py`, `train.py`, `requirements.txt`
3. The Space should automatically build and deploy
4. Once built, you can test the app in the web interface

## Important Notes

- **Model Loading**: The app automatically loads from `Sualeh77/smollm2-135m-trained-on-tinyShakespear-forfun` model repo
- **Checkpoint**: Uses `smollm2-step=05000-train_loss=0.0918.ckpt`
- **First Load**: The first time the Space loads, it will download the checkpoint from the model repo (may take a few minutes)
- **Caching**: Subsequent loads will be faster due to Hugging Face caching

## Troubleshooting

### If push fails with "non-fast-forward":
```bash
# Fetch latest
git fetch huggingface

# Rebase (without GPG signing)
git config --global commit.gpgsign false
git rebase huggingface/main
git push huggingface main
git config --global commit.gpgsign true
```

### If Space build fails:
- Check the "Logs" tab in your Space
- Ensure all dependencies are in `requirements.txt`
- Make sure `app.py` is the entry point (it should be automatically detected)

### If model loading fails:
- Verify the model repo name is correct: `Sualeh77/smollm2-135m-trained-on-tinyShakespear-forfun`
- Verify the checkpoint name: `smollm2-step=05000-train_loss=0.0918.ckpt`
- Check that the checkpoint file exists in the model repo
