import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import wave

from src.mute.model.mossformer2 import Mossformer2Wrapper

model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/mossformer2-librimix-2spk')


def voice_indexer(path, top):
    zaza, sample_rate = model.reformer(path)
    voice = {}
    lendo = 0
    voice_len = int(zaza.shape[1] / 8) + 1
    for i in range(zaza.shape[2]):
        index_seconds = []
        index = librosa.effects.split(zaza[0][..., i], top_db=top)
        index_seconds = (index / 8000).tolist()
        voice[i] = index_seconds
    # mute_percent = max(0,1-(lendo/zaza.shape[1]))
    mute_percent = min(1, (lendo / zaza.shape[1]))
    return voice, voice_len


# shows the sound waves
def visualize(path: str, mask_range: dict, name: str):
    raw = wave.open(path)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")
    f_rate = raw.getframerate()
    time = np.linspace(
        0,  # start
        len(signal) / f_rate,
        num=len(signal)
    )
    plt.figure(figsize=(20, 4))  # (10, 2))
    plt.title(f"Sound Wave: {name}")
    plt.xlabel("Time")
    plt.plot(time, signal)
    colors = ['red', 'green', 'yellow', 'cyan', 'magenta', 'blue', 'black', 'white']
    for keys in mask_range.keys():
        for val in mask_range[keys]:
            plt.axvspan(val[0], val[1], color=colors[keys], alpha=0.2)
    # plt.show()
    plt.savefig(f"{name}.jpg")


def percenter(path: str, mask_range: dict, voice_len: int):
    sigma = np.zeros(shape=voice_len, dtype=int)
    for keys in mask_range.keys():
        for val in mask_range[keys]:
            x = int(val[0] * 1000)
            y = int(val[1] * 1000)
            for i in range(x, y):
                sigma[i] = 1
    return (np.sum(sigma) / voice_len)


def percentage_mute_finder(path, decibel_sensitivity=15, mute_sensitivity=0.5):
    try:
        zaza, sample_rate = model.reformer(path)
        voice = {}
        lendo = 0
        voice_len = int(zaza.shape[1] / 8) + 1
        for i in range(zaza.shape[2]):
            index_seconds = []
            index = librosa.effects.split(zaza[0][..., i], top_db=decibel_sensitivity)
            index_seconds = (index / 8000).tolist()
            voice[i] = index_seconds

        sigma = np.zeros(shape=voice_len, dtype=int)
        zetta = np.zeros(shape=voice_len, dtype=int)
        omega = []
        alpha = []
        for keys in voice.keys():
            for val in voice[keys]:
                x = int(val[0] * 1000)
                y = int(val[1] * 1000)
                if (y - x) / 1000 > mute_sensitivity:
                    omega.append(val)
                for i in range(x, y):
                    sigma[i] = 1
        m, n, k = 0, 0, 0
        while k < len(sigma):
            if sigma[k] == 0:
                m = k
                n = k
                while sigma[n] == 0 and n < len(sigma) - 1:
                    n += 1
                k = n
                alpha.append([m / 1000, n / 1000])
            k += 1

        for val in alpha:
            x = int(val[0] * 1000)
            y = int(val[1] * 1000)
            if (y - x) / 1000 > mute_sensitivity:
                # omega.append(val)
                for i in range(x, y):
                    zetta[i] = 1

        return f"{1 - (np.sum(sigma) / voice_len)}", omega
    except:
        return "Mute Percentage can't be calculated."


percentage_mute_finder('/home/makhataei/Projects/stt/test/QC/4624562.wav', decibel_sensitivity=30, mute_sensitivity=5)

PATH = "/home/makhataei/Projects/stt/test/QC"
a = os.listdir(PATH)
plt.close('all')
for file in a:
    for top in [5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60]:  # for top in [30]:#
        rep, voice_len = voice_indexer(f"{PATH}/{file}", top)
        # with open("indexer.csv", "a") as fileee:
        #     fileee.writelines(
        #         f"{file} - {top}db: \n \t {rep} \n"
        #     )
        # visualize(path=f"{PATH}/{file}",mask_range=rep,name=f"{file} in {top}db")
        # b= percenter(path=f"{PATH}/{file}",mask_range=rep,voice_len=voice_len)
        aaaa = percentage_mute_finder(path=f"{PATH}/{file}", decibel_sensitivity=top)
        print("yay")
