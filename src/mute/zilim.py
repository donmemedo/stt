import gc
import librosa
import numpy as np
import os
import torch

from src.config import whispers
from src.mute.model.mossformer2 import Mossformer2Wrapper

model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/mossformer2-librimix-2spk')

from src.gender.gender_classifier import gender_predict


def dupi(path):
    try:
        # sound_array, sample_rate = model.reformer(path)
        new_path = path.split('.')[0]
        if not os.path.exists(f'{new_path}'):
            os.makedirs(f'{new_path}')

        model.inference(path, f'{new_path}')
        gc.collect()
        torch.cuda.empty_cache()
        return os.listdir(f'{new_path}')
    except:
        return 0


path = "/media/makhataei/Backups/Projects/STT/test/100 Calls/4527627.wav"
bebe = dupi(path)
patho = path.split('.')[0]
for path in bebe:
    print(gender_predict(f"{patho}/{path}"))

print("yay")


def percentage_mute_finder(path, mo_decibel_sensitivity=1, mute_sensitivity=0.5, tone_decibel_sensitivity=15,
                           tone_sensitivity=0.01, overlap_sensitivity=0):
    try:
        sound_array, sample_rate = model.reformer(path)
        voice = {}
        lendo = 0
        voice_len = int(sound_array.shape[1] / 8) + 1
        overlap_detector = np.zeros(shape=int(sound_array.shape[1]) + 1, dtype=int)
        list_of_overlaps = []
        list_of_overlaps_with_sensitivity = []
        for i in range(sound_array.shape[2]):
            index_seconds = []
            index = librosa.effects.split(sound_array[0][..., i], top_db=tone_decibel_sensitivity)
            # for tito in index:
            #     for t in range(tito[0], tito[1]):
            #         overlap_detector[t] += 1
            index_seconds = (index / 8000).tolist()
            voice[i] = index_seconds
        tone_detector = np.zeros(shape=voice_len, dtype=int)
        tone_detector_with_sensitivity = np.zeros(shape=voice_len, dtype=int)
        list_of_tone = []
        list_of_tone_with_sensitivity = []
        time_of_tone = 0
        for keys in voice.keys():
            for val in voice[keys]:
                x = int(val[0] * 1000)
                y = int(val[1] * 1000)
                if (y - x) / 1000 > tone_sensitivity:
                    time_of_tone = time_of_tone + y - x
                for i in range(x, y):
                    tone_detector[i] = 1
        m, n, k = 0, 0, 0
        while k < len(tone_detector):
            if tone_detector[k] == 0:
                m = k
                n = k
                while tone_detector[n] == 0 and n < len(tone_detector) - 1:
                    n += 1
                k = n
                list_of_tone.append([m / 1000, n / 1000])
            k += 1
        for val in list_of_tone:
            x = int(val[0] * 1000)
            y = int(val[1] * 1000)
            if (y - x) / 1000 > tone_sensitivity:
                list_of_tone_with_sensitivity.append(val)
                for i in range(x, y):
                    tone_detector_with_sensitivity[i] = 1

        m, n, k = 0, 0, 0
        while k < len(tone_detector_with_sensitivity):
            if tone_detector_with_sensitivity[k] == 0:
                m = k
                n = k
                while tone_detector_with_sensitivity[n] == 0 and n < len(tone_detector_with_sensitivity) - 1:
                    n += 1
                k = n
            k += 1
        for i in range(sound_array.shape[2]):
            index_seconds = []
            index = librosa.effects.split(sound_array[0][..., i], top_db=mo_decibel_sensitivity)
            for tito in index:
                for t in range(tito[0], tito[1]):
                    overlap_detector[t] += 1
            index_seconds = (index / 8000).tolist()
            voice[i] = index_seconds
        m, n, k = 0, 0, 0
        while k < len(overlap_detector):
            if overlap_detector[k] == 2:
                m = k
                n = k
                while overlap_detector[n] == 2 and n < len(overlap_detector) - 1:
                    n += 1
                k = n
                list_of_overlaps.append([m / 8000, n / 8000])
            k += 1
        for val in list_of_overlaps:
            x = int(val[0] * 1000)
            y = int(val[1] * 1000)
            if (y - x) / 1000 > overlap_sensitivity:
                list_of_overlaps_with_sensitivity.append(val)
        mute_detector = np.zeros(shape=voice_len, dtype=int)
        mute_detector_with_sensitivity = np.zeros(shape=voice_len, dtype=int)
        list_of_mute = []
        list_of_mute_with_sensitivity = []
        time_of_mute = 0
        for keys in voice.keys():
            for val in voice[keys]:
                x = int(val[0] * 1000)
                y = int(val[1] * 1000)
                if (y - x) / 1000 > mute_sensitivity:
                    time_of_mute = time_of_mute + y - x
                for i in range(x, y):
                    mute_detector[i] = 1
        m, n, k = 0, 0, 0
        while k < len(mute_detector):
            if mute_detector[k] == 0:
                m = k
                n = k
                while mute_detector[n] == 0 and n < len(mute_detector) - 1:
                    n += 1
                k = n
                list_of_mute.append([m / 1000, n / 1000])
            k += 1
        for val in list_of_mute:
            x = int(val[0] * 1000)
            y = int(val[1] * 1000)
            if (y - x) / 1000 > mute_sensitivity:
                list_of_mute_with_sensitivity.append(val)
                for i in range(x, y):
                    mute_detector_with_sensitivity[i] = 1

        m, n, k = 0, 0, 0
        while k < len(mute_detector_with_sensitivity):
            if mute_detector_with_sensitivity[k] == 0:
                m = k
                n = k
                while mute_detector_with_sensitivity[n] == 0 and n < len(mute_detector_with_sensitivity) - 1:
                    n += 1
                k = n
            k += 1
        return (200, f"{np.sum(mute_detector) / voice_len}", list_of_mute, np.sum(
            mute_detector_with_sensitivity) / voice_len, list_of_mute_with_sensitivity,
                list_of_overlaps_with_sensitivity, f"{np.sum(tone_detector) / voice_len}", list_of_tone, np.sum(
            tone_detector_with_sensitivity) / voice_len, list_of_tone_with_sensitivity)

    except:
        return 400, "Can't be calculated.", 0, 0, 0, 0, 0, 0, 0, 0


def trancripter(path, transcriber):
    try:
        sound_array, sample_rate = model.reformer(path)
        speech_texts = {}
        for i in range(sound_array.shape[2]):
            speech_texts[i] = transcriber(sound_array[0][..., i])
        return 200, speech_texts
    except:
        return 400, {0: "", 1: ""}
