from huggingface_hub import login
login(
    token="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",
    add_to_git_credential=True,
    new_session=True,
    write_permission=True,
)


from datasets import load_dataset

minds_14 = load_dataset("KELONMYOSA/dusha_emotion_audio") # for French
# to download all data for multi-lingual fine-tuning uncomment following line
# minds_14 = load_dataset("PolyAI/all", "all")

# see structure
print(minds_14)

# minds_14 = minds_14.train_test_split(test_size=0.2)
minds_14 = minds_14.remove_columns(["file"])#, "transcription", "english_transcription", "lang_id"])
minds_14["train"][0]

labels = minds_14["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

from datasets import load_dataset, Audio
minds_14 = minds_14.cast_column("audio", Audio(sampling_rate=16_000))
minds_14["train"][0]

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs
encoded_minds = minds_14.map(preprocess_function, remove_columns="audio", batched=True)
# encoded_minds = encoded_minds.rename_column("intent_class", "label")

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
   "facebook/wav2vec2-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

training_args = TrainingArguments(
    "my_awesome_mind_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()