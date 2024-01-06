---
language:
- fa
license: apache-2.0
base_model: makhataei/Whisper-Small-Common-Voice
tags:
- fa-asr
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_16_0
metrics:
- wer
model-index:
- name: Whisper Small Persian
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Whisper Small Persian

This model is a fine-tuned version of [makhataei/Whisper-Small-Common-Voice](https://huggingface.co/makhataei/Whisper-Small-Common-Voice) on the Common Voice 16.0 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7215
- Wer: 43.6054

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-07
- train_batch_size: 10
- eval_batch_size: 10
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 40
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 50
- training_steps: 3000

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer     |
|:-------------:|:-----:|:----:|:---------------:|:-------:|
| 0.0006        | 0.16  | 100  | 0.7099          | 42.5759 |
| 0.0016        | 0.31  | 200  | 0.7139          | 42.6710 |
| 0.0051        | 0.47  | 300  | 0.7114          | 42.6803 |
| 0.004         | 0.62  | 400  | 0.7109          | 42.6130 |
| 0.0047        | 0.78  | 500  | 0.7101          | 42.3139 |
| 0.0029        | 0.93  | 600  | 0.7105          | 42.4113 |
| 0.0016        | 1.09  | 700  | 0.7101          | 42.4067 |
| 0.0016        | 1.25  | 800  | 0.7129          | 42.3997 |
| 0.0028        | 1.4   | 900  | 0.7145          | 42.3951 |
| 0.0019        | 1.56  | 1000 | 0.7135          | 42.3557 |
| 0.0019        | 1.71  | 1100 | 0.7132          | 42.3534 |
| 0.0021        | 1.87  | 1200 | 0.7128          | 42.4739 |
| 0.0023        | 2.03  | 1300 | 0.7144          | 42.9330 |
| 0.0021        | 2.18  | 1400 | 0.7151          | 42.8403 |
| 0.0019        | 2.34  | 1500 | 0.7158          | 42.8820 |
| 0.0014        | 2.49  | 1600 | 0.7166          | 42.9121 |
| 0.0262        | 2.65  | 1700 | 0.7171          | 42.9933 |
| 0.0022        | 2.8   | 1800 | 0.7183          | 42.9608 |
| 0.0012        | 2.96  | 1900 | 0.7185          | 43.3921 |
| 0.0018        | 3.12  | 2000 | 0.7187          | 43.3295 |
| 0.0015        | 3.27  | 2100 | 0.7191          | 43.3480 |
| 0.0013        | 3.43  | 2200 | 0.7198          | 43.3898 |
| 0.0019        | 3.58  | 2300 | 0.7200          | 43.4106 |
| 0.0017        | 3.74  | 2400 | 0.7206          | 43.4315 |
| 0.0015        | 3.9   | 2500 | 0.7209          | 43.4268 |
| 0.0016        | 4.05  | 2600 | 0.7210          | 43.6031 |
| 0.0267        | 4.21  | 2700 | 0.7212          | 43.5891 |
| 0.001         | 4.36  | 2800 | 0.7213          | 43.5938 |
| 0.0015        | 4.52  | 2900 | 0.7215          | 43.6031 |
| 0.0019        | 4.67  | 3000 | 0.7215          | 43.6054 |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.0.1+cu117
- Datasets 2.15.0
- Tokenizers 0.15.0
