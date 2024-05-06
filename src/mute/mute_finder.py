from model.mossformer2 import Mossformer2Wrapper
import librosa


def mute_finder(path,decibel_sensitivity=15):
    model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/mossformer2-librimix-2spk')
    try:
        zaza,sample_rate = model.reformer(path)
        lendo =0
        for i in range(zaza.shape[2]):
            index = librosa.effects.split(zaza[0][..., i], top_db=decibel_sensitivity)
            yt = librosa.effects.remix(zaza[0][..., i], index)
            lendo +=len(yt)
        mute_percent = max(0,1-(lendo/zaza.shape[1]))
        return f"Mute Percentage is: \t{mute_percent}\n"
    except:
        return "Mute Percentage can't be calculated."
