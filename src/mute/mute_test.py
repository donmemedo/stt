# from mute_finder import percentage_mute_finder
import pandas as pd
import os
from src.mute.model.mossformer2 import Mossformer2Wrapper
import librosa

# PATH = "/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/inputs"
# a = os.listdir(PATH)
#
# for file in a:
#     try:
#         bb1 = percentage_mute_finder(f"{PATH}/{file}",1.25)
#         bb2 = percentage_mute_finder(f"{PATH}/{file}", 2.5)
#         bb3 = percentage_mute_finder(f"{PATH}/{file}", 5)
#         bb4 = percentage_mute_finder(f"{PATH}/{file}", 10)
#         with open("mute.csv", "a") as fileee:
#             fileee.writelines(
#             f"{file}, {bb1},{bb2},{bb3},{bb4}\n"
#             )
#     except:
#         pass
#
#
path = "/home/makhataei/Projects/stt/test/QC/4539195.wav"
model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/mossformer2-librimix-2spk')


def voice_indexer(path,top):
    zaza, sample_rate = model.reformer(path)
    voice = {}
    for i in range(zaza.shape[2]):
        # index_seconds = []
        index = librosa.effects.split(zaza[0][..., i], top_db=top)
        index_seconds = (index / 8000).tolist()
        voice[i] = index_seconds
    return voice
# a= voice_indexer(path,10)


PATH="/home/makhataei/Projects/stt/test/QC"
a = os.listdir(PATH)

for file in a:
    for top in [5,10,20,40,60]:
        rep=voice_indexer(f"{PATH}/{file}",top)
        with open("indexer.csv", "a") as fileee:
            fileee.writelines(
                f"{file} - {top}db: \n \t {rep} \n"
            )
