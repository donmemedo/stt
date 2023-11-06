from datasets import Dataset
import pandas as pd
import librosa
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate


## we will load the both of the data here.
train1_df = pd.read_csv("../datasets/Common_Voice_Corpus_15/fa/train1.csv")
train2_df = pd.read_csv("../datasets/Common_Voice_Corpus_15/fa/train2.csv")
test_df = pd.read_csv("../datasets/Common_Voice_Corpus_15/fa/test.csv")
PATH = "/home/makhataei/Projects/STT/datasets/Common_Voice_Corpus_15/fa/clips/"

audio11 = []
for i in list(train1_df.path):
    audio11.append(librosa.load(path=PATH + i, sr=16000)[0])
train1_df["audio"] = audio11

audio12 = []
for i in list(train2_df.path):
    audio12.append(librosa.load(path=PATH + i, sr=16000)[0])
train2_df["audio"] = audio12

audio2 = []
for i in list(test_df.path):
    audio2.append(librosa.load(path=PATH + i, sr=16000)[0])
test_df["audio"] = audio2

train1_dataset = Dataset.from_pandas(train1_df)
train2_dataset = Dataset.from_pandas(train2_df)
test_dataset = Dataset.from_pandas(test_df)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

## Load WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="Persian", task="transcribe"
)

## Combine To Create A WhisperProcessor
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="Persian", task="transcribe"
)


def prepare_dataset(examples):
    # compute log-Mel input features from input audio array
    del examples["accents"]
    del examples["age"]
    del examples["client_id"]
    del examples["down_votes"]
    del examples["gender"]
    del examples["locale"]
    del examples["path"]
    del examples["segment"]
    del examples["up_votes"]

    audio = examples["audio"]
    examples["input_features"] = feature_extractor(
        audio, sampling_rate=16000, max_length=480000
    ).input_features[0]
    del examples["audio"]
    sentences = examples["sentence"]
    # encode target text to label ids
    examples["labels"] = tokenizer(sentences, truncation=True, max_length=448).input_ids
    del examples["sentence"]
    return examples


train1_dataset = train1_dataset.map(prepare_dataset, num_proc=1)
train2_dataset = train2_dataset.map(prepare_dataset, num_proc=1)
test_dataset = test_dataset.map(prepare_dataset, num_proc=1)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it’s append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Define the Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./Whisper-small-Mozilla",
    per_device_train_batch_size=14,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=10,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer1 = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train1_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print("Train Phase1 Started.")
trainer1.train()

trainer2 = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train2_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print("Train Phase2 Started.")
trainer2.train()
