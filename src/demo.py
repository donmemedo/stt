from transformers import pipeline
import gradio as gr

MODEL = "../models/checkpoint-1100/config.json"
AUDIO = "../test/Tehran_Namaesh0_5.wav"
import time
import whisper

my_file = open("transcript.txt",'a') 

for AUDIO in [
    "../test/common_voice_fa_18558703.mp3",
    "../test/common_voice_fa_19262884.mp3",
    "../test/common_voice_fa_19372453.mp3",
    "../test/common_voice_fa_19533858.mp3",
    "../test/common_voice_fa_25126597.mp3",
    "../test/common_voice_fa_27883343.mp3",
    "../test/common_voice_fa_27979630.mp3",
    "../test/common_voice_fa_29377504.mp3",
    "../test/common_voice_fa_29426046.mp3",
    "../test/common_voice_fa_30663453.mp3",
    "../test/common_voice_fa_31493167.mp3",
    "../test/common_voice_fa_33138107.mp3",
]:
    print(AUDIO,"-------------------- \n")
    for typo in ["cuda", "cpu"]:
        for size in ["base", "small", "medium", "large-v1", "large-v2"]:
            start = int(1000*time.time())
            try:
                model = whisper.load_model(name=size, device=typo)
            except:
                break
            result = model.transcribe(AUDIO)
            end = int(1000*time.time())
            my_file.writelines("---------------------\n\n")
            my_file.writelines(f"inference time for {typo} in {size}-Model is {end-start} milliseconds: \n")
            my_file.writelines(result["text"])
            my_file.writelines("\n\n ---------------------")
            print(
                f"inference time for {typo} in {size}-Model is {end-start} milliseconds: \n"
            )
            print(result["text"])
my_file.close()
