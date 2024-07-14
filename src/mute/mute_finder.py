from src.mute.model.mossformer2 import Mossformer2Wrapper
import librosa
import numpy as np

model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/mossformer2-librimix-2spk')


def percentage_mute_finder(path, decibel_sensitivity=15, mute_sensitivity=0.5, overlap_sensitivity=0):
    try:
        zaza, sample_rate = model.reformer(path)
        voice = {}
        lendo = 0
        voice_len = int(zaza.shape[1] / 8) + 1
        delta = np.zeros(shape=int(zaza.shape[1]) + 1, dtype=int)
        gamma = []
        theta = []
        for i in range(zaza.shape[2]):
            index_seconds = []
            index = librosa.effects.split(zaza[0][..., i], top_db=decibel_sensitivity)
            for tito in index:
                for t in range(tito[0], tito[1]):
                    delta[t] += 1
            index_seconds = (index / 8000).tolist()
            voice[i] = index_seconds

        m, n, k = 0, 0, 0
        while k < len(delta):
            if delta[k] == 2:
                m = k
                n = k
                while delta[n] == 2 and n < len(delta) - 1:
                    n += 1
                k = n
                gamma.append([m / 8000, n / 8000])
            k += 1
        for val in gamma:
            x = int(val[0] * 1000)
            y = int(val[1] * 1000)
            if (y - x) / 1000 > overlap_sensitivity:
                theta.append(val)
        sigma = np.zeros(shape=voice_len, dtype=int)
        zetta = np.zeros(shape=voice_len, dtype=int)
        alpha = []
        micro = []
        beta = 0
        for keys in voice.keys():
            for val in voice[keys]:
                x = int(val[0] * 1000)
                y = int(val[1] * 1000)
                if (y - x) / 1000 > mute_sensitivity:
                    beta = beta + y - x
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
                micro.append(val)
                for i in range(x, y):
                    zetta[i] = 1

        m, n, k = 0, 0, 0
        while k < len(zetta):
            if zetta[k] == 0:
                m = k
                n = k
                while zetta[n] == 0 and n < len(zetta) - 1:
                    n += 1
                k = n
            k += 1
        return 200, f"{np.sum(sigma) / voice_len}", alpha, np.sum(zetta) / voice_len, micro, theta
    except:
        return 400, "Mute Percentage can't be calculated.", 0, 0, 0
