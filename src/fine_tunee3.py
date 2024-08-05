# import cudf.pandas
# cudf.pandas.install()
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

from huggingface_hub import interpreter_login, login

# interpreter_login()
login(
    token="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",
    add_to_git_credential=True,
    new_session=True,
    write_permission=True,
)
## we will load the both of the data here.
train_df = pd.read_csv(
    "../datasets/Common_Voice_Corpus_16/fa/train.csv", on_bad_lines="skip"
)
test_df = pd.read_csv(
    "../datasets/Common_Voice_Corpus_16/fa/test.csv", on_bad_lines="skip"
)
PATH = "/home/makhataei/Projects/STT/datasets/Common_Voice_Corpus_16/fa/clips/"

audio1 = []
for i in list(train_df.path):
    audio1.append(librosa.load(path=PATH + i, sr=16000)[0])
train_df["audio"] = audio1

audio2 = []
for i in list(test_df.path):
    audio2.append(librosa.load(path=PATH + i, sr=16000)[0])
test_df["audio"] = audio2

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "makhataei/Whisper-Small-Common-Voice"
)

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


train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
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


model = WhisperForConditionalGeneration.from_pretrained(
    "makhataei/Whisper-Small-Common-Voice"
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Define the Training Arguments
training_args = Seq2SeqTrainingArguments(
    f"Whisper-Small-Common-Voice",
    # output_dir="/media/makhataei/Backups/Whisper-Small-Common-Voice",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=4,
    learning_rate=1e-7,
    warmup_steps=50,
    max_steps=3000,
    gradient_checkpointing=True,
    # fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=10,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    save_total_limit=3,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print("FineTune Phase Started.")
trainer.train()
trainer.create_model_card(
    language="fa",
    tags="fa-asr",
    model_name="Whisper Small Persian",
    finetuned_from="makhataei/Whisper-Small-Common-Voice",
    tasks="transcribe",
    dataset_tags="mozilla-foundation/common_voice_16_0",
    dataset="Common Voice 16.0",
    dataset_args="config: fa, split: train,test",
)
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_16_0",
    "dataset": "Common Voice 16.0",
    "dataset_args": "config: fa, split: train,test",
    "language": "fa",
    "model_name": "Whisper Small Persian",
    "finetuned_from": "makhataei/Whisper-Small-Common-Voice",
    "tasks": "transcribe",
    "tags": "fa-asr",
}
trainer.push_to_hub(**kwargs)
