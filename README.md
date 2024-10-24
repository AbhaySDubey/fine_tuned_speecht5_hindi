# Hindi Text-to-Speech Finetuning Project

This project demonstrates how to finetune Microsoft's SpeechT5 model for Hindi Text-to-Speech (TTS) synthesis using the Mozilla Common Voice dataset.

## Overview

The project finetunes the SpeechT5 model to generate natural-sounding Hindi speech from text input. It uses the Mozilla Common Voice Hindi dataset for training and implements custom text normalization for Hindi characters.

## Prerequisites

- Python 3.x
- NVIDIA GPU with CUDA support
- Google Colab (recommended) or similar environment
- Hugging Face account

## Required Libraries

```bash
pip install transformers datasets soundfile speechbrain accelerate
```

## Project Structure

1. Data Preparation
   - Loading Mozilla Common Voice Hindi dataset
   - Audio preprocessing (16kHz sampling rate)
   - Text normalization for Hindi characters

2. Model Components
   - SpeechT5 base model (microsoft/speecht5_tts)
   - SpeechT5 processor and tokenizer
   - SpeechBrain speaker embeddings
   - HiFi-GAN vocoder


## Key Features

- Custom Hindi character support
- Speaker embedding integration
- Efficient batch processing
- Mixed precision training
- Gradient checkpointing for memory efficiency

## Training Details

- Batch size: 4
- Gradient accumulation steps: 8
- Learning rate: 1e-5
- Warmup steps: 500
- Maximum steps: 4000
- FP16 training enabled
- Evaluation every 1000 steps
- Model checkpointing every 1000 steps

## Model Output

The finetuned model is available on Hugging Face Hub at `intuitive262/speecht5_finetuned_for_hindi_tts`.

## Usage Example

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch

# Load models
processor = SpeechT5Processor.from_pretrained("intuitive262/speecht5_finetuned_for_hindi_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("intuitive262/speecht5_finetuned_for_hindi_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Prepare input text
text = "आप कैसे हैं?"  # "How are you?" in Hindi
inputs = processor(text=text, return_tensors="pt")

# Generate speech
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
```
