# STT
Speech To Text Full Project

## Stages
This **Project** has these stages:

### 1. Whisper Project:
This Stage separates Voices of *Customer* and *Agent* from each other to transcript it easier.

### 2. FairSeq Project:
This Stage Transcripts everything that heard from ***One Person***.

##### 2.1 Pre:
Requirements:

Python 3.10
```shell
mkdir -p STT && cd STT
mkdir -p models && mkdir -p tests 
mkdir -p src && cd src
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable .
/usr/local/bin/python3.10 ./STT/src/fairseq/examples/mms/asr/infer/mms_infer.py --model "./STT/models/MODEL.pt" --lang fas --audio "./STT/test/AUDIO.wav"
```


### 3. Semantic Handlers:
It's a small-size NLP Layer to check semantic of produced words.