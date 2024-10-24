!nvidia-smi

!pip install transformers datasets soundfile speechbrain accelerate

from huggingface_hub import notebook_login
notebook_login()

"""##### Mount Google Drive"""

from google.colab import drive
drive.mount('/content/drive')

"""#### Directory Path"""

dir_path = '/content/drive/MyDrive/hindi_tts_finetuning'

import os

os.makedirs(dir_path, exist_ok=True)
print(f"Directory created or already exists: {dir_path}")

from datasets import load_dataset, Audio

dataset = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train")
len(dataset)

dataset[0]

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

"""### Data pre-processing"""

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)

tokenizer = processor.tokenizer

"""### Text Normalization
#### 1. Extract all characters
#### 2. Compare with tokenized(identified) characters
#### 3. Add all the unidentified characters to the tokenizer as new tokens and save it
"""

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs['vocab'][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

unidentified_vocab = dataset_vocab - tokenizer_vocab
for char in unidentified_vocab:
  print(char, end=", ")

hindi_tokens = ['ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ऑ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', '़', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॅ', 'े', 'ै', 'ॉ', 'ो', 'ौ', '्', 'क़', 'ज़', 'ड़', 'ढ़', 'फ़',]
tokenizer.add_tokens(hindi_tokens)

tokenizer.save_pretrained(f'{dir_path}/tokenizer')

model.resize_token_embeddings(len(tokenizer))

"""##### Check if tokenizer can recognize Hindi vocab now"""

tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

dataset_vocab - tokenizer_vocab

import re

def remove_punctuation(text):
    # Regex pattern for Hindi punctuation marks
    hindi_punctuation_pattern = r'[।|,;:!?(){}[\]<>।॥"\'“”‘’—-]'
    cleaned_text = re.sub(hindi_punctuation_pattern, '', text)
    return cleaned_text

def apply_cleaning(batch):
  batch["sentence"] = [remove_punctuation(sentence) for sentence in batch["sentence"]]
  return batch

dataset = dataset.map(apply_cleaning, batched=True, batch_size=100)

dataset[0]

def tokenize_hindi(text):
  tokens = tokenizer.tokenize(text)
  return tokens

for i in range(min(5, len(dataset))):
  sentence = dataset[i]["sentence"]
  print(f"Cleaned: {sentence}")
  tokens = tokenize_hindi(sentence)
  print(f"Tokens: {tokens}\n")

from collections import defaultdict

speaker_counts = defaultdict(int)

for client_id in dataset["client_id"]:
  speaker_counts[client_id] += 1

speaker_counts

import matplotlib.pyplot as plt

plt.figure()
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Samples")
plt.show()

"""I'm selecting all the speakers and all the samples, as the sample size itself is pretty small and there are only 3 unique speakers"""

import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=f'{dir_path}/speaker_embeddings',
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["sentence"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

processed_example = prepare_dataset(dataset[0])
list(processed_example.keys())

processed_example["speaker_embeddings"].shape

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(processed_example["labels"].T)
plt.show()

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

"""### Data Collator"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

"""### Training the model"""

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="intuitive262/speecht5_finetuned_for_hindi_tts",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

trainer.train()

trainer.push_to_hub()

"""### Inference"""

new_model = SpeechT5ForTextToSpeech.from_pretrained("intuitive262/speecht5_finetuned_for_hindi_tts")

example = dataset["test"][1]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech = new_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

from IPython.display import Audio

Audio(speech.numpy(), rate=16000)

