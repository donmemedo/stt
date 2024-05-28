from src.mute.model.mossformer2 import Mossformer2Wrapper
import librosa
import numpy as np

model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/mossformer2-librimix-2spk')
def percentage_mute_finder(path,decibel_sensitivity=15):
    try:
        zaza, sample_rate = model.reformer(path)
        voice = {}
        lendo = 0
        voice_len = int(zaza.shape[1]/8)+1
        for i in range(zaza.shape[2]):
            index_seconds = []
            index = librosa.effects.split(zaza[0][..., i], top_db=decibel_sensitivity)
            index_seconds = (index / 8000).tolist()
            voice[i] = index_seconds

        sigma = np.zeros(shape=voice_len,dtype=int)
        for keys in voice.keys():
            for val in voice[keys]:
                x= int(val[0]*1000)
                y= int(val[1]*1000)
                for i in range(x,y):
                    sigma[i]=1
        return f"{1-(np.sum(sigma)/voice_len)}"
    except:
        return "Mute Percentage can't be calculated."
