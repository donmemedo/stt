import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd

SR=8000


def mute_finder(path,filename,samplerate,decibels):
	y, sr = librosa.core.load(f'{path}/{filename}', sr=samplerate)
	index = librosa.effects.split(y, top_db=decibels)
	yt = librosa.effects.remix(y, index)
	print(yt)


# PATH='/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/dumbeldor'
# a = os.listdir(PATH)
# for folder in a:
# 	paths = f'{PATH}/{folder}'
# 	b = os.listdir(paths)
# 	for file in b:
# 		try:
# 			mute_finder(paths,file,SR,20)
# 			mute_finder(paths,file,SR,15)
#
# 		except:
# 			pass
#
mute_finder('/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/dumbeldor/4527615.wav/','index1.wav',SR,15)
mute_finder('/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/dumbeldor/4527615.wav/','index1.wav',SR,20)
