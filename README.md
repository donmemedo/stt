# STT
Speech To Text Full Project

## Stages
This **Project** has these stages:

### 1. Whisper Project:
This Stage separates Voices of *Customer* and *Agent* from each other to transcript it easier.
```shell
pip install torch==2.0.1 torchaudio torchvision
pip install -U openai-whisper
cd src/tests
whisper AUDIO.wav --model MODEL --language Persian
```

### 2. FairSeq Project:
This Stage Transcripts everything that heard from ***One Person***.
```shell
cd src
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable .
/usr/local/bin/python3.10 ./STT/src/fairseq/examples/mms/asr/infer/mms_infer.py --model "./STT/models/MODEL.pt" --lang fas --audio "./STT/test/AUDIO.wav"
```


##### 2.1 Pre:
Requirements:
```shell
mkdir -p STT && cd STT
mkdir -p models && mkdir -p tests 
mkdir -p src && cd src
python3.10 -m pip install -r requirements.txt
```
Python 3.10


### 3. Semantic Handlers:
It's a small-size NLP Layer to check semantic of produced words.