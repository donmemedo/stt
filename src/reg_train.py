# from src.ser.inference import prediction
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import pandas as pd
from datasets import Dataset

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

PATH = "/media/makhataei/Backups/555/"  # /home/makhataei/Projects/CallCanterQC/datasets/DushaEmotionAudio"
train_df = pd.read_csv("/home/makhataei/Projects/CallCanterQC/src/ser/tester.csv", on_bad_lines='skip')
a = train_df.values
#
# for file in a:
#     with open(f'testoor.csv', 'a') as fileee:
#         fileee.writelines(
#             # f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n')
#             f'{file[0]},{np.random.random()*100}\n')

import torch, gc, random, datasets
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

model_name = "anantoj/wav2vec2-adult-child-cls"
model_name = "facebook/wav2vec2-base"

df = pd.read_csv("/home/makhataei/Projects/CallCanterQC/src/ser/testoor.csv", on_bad_lines="skip")
msk = np.random.rand(len(df)) < 0.9
train_df = df[msk]
test_df = df[~msk]

# train_df = pd.read_csv(
#     "/home/makhataei/Projects/CallCanterQC/src/ser/testoor.csv", on_bad_lines="skip"
# )
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

# Make data
# X = train_df.audio
# y = train_df.label
#
# # Split Data
# X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y, test_size=0.1)
# z_train, z_test= train_test_split(train_df.tolist(), test_size=0.1)

# Call the Tokenizer
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


def prepare_dataset(examples):
    # compute log-Mel input features from input audio array
    audio = examples["audio"]
    examples["input_values"] = feature_extractor(
        audio, sampling_rate=16000, max_length=480000
    ).data['input_values']  # .input_features[0]
    del examples["audio"]
    # sentences = examples["sentence"]
    # # encode target text to label ids
    # examples["labels"] = tokenizer(sentences, truncation=True, max_length=448).input_ids
    # del examples["sentence"]
    return examples


train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
test_dataset = test_dataset.map(prepare_dataset, num_proc=1)


def preprocess_function(examples):
    # audio_arrays = [x["array"] for x in examples["audio"]]
    # audio_arrays = [examples["audio"]]
    inputs = feature_extractor(
        examples,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * 30),
        truncation=True,
    )
    return inputs


# train_encodings = preprocess_function(X_train)
# valid_encodings = preprocess_function(X_test)

# Encode the text
# train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
# valid_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)


class MakeTorchData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        # item["labels"] = float(item["labels"])
        item["labels"] = (item["labels"]).long()
        return item

    def __len__(self):
        return len(self.labels)


# convert our tokenized data into a torch Dataset
# train_dataset = MakeTorchData(train_encodings, y_train.ravel())
# valid_dataset = MakeTorchData(valid_encodings, y_test.ravel())
# train_dataset = MakeTorchData(train_encodings, y_train.ravel())
# valid_dataset = MakeTorchData(valid_encodings, y_test.ravel())

model = AutoModelForAudioClassification.from_pretrained(model_name,
                                                        num_labels=1).to("cuda")


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1 / len(labels) * np.sum(2 * np.abs(logits - labels) / (np.abs(labels) + np.abs(logits)) * 100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


# Specifiy the arguments for the trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir='./logs',
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model='rmse',
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Call the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics_for_regression,
)

# Train the model
trainer.train()

# Call the summary
trainer.evaluate()
