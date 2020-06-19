# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from os import path
from pydub import AudioSegment
import librosa
#from entropy import pectral_entropy


# files
src = "./Yash.mp3"
dst = "test.wav"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")

fs_rate, signal = wavfile.read("test.wav")
samples, sample_rate = librosa.load("test.wav")
print ("Frequency sampling", fs_rate)
print(signal)

spec = np.abs(np.fft.rfft(signal))
freq = np.fft.rfftfreq(len(signal), d=1 / fs_rate)
print(freq)
spec = np.abs(spec)
amp = spec / spec.sum()
mean = (freq * amp).sum()
sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
amp_cumsum = np.cumsum(amp)
median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
mode = freq[amp.argmax()]
Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
IQR = Q75 - Q25
z = amp - amp.mean()
w = amp.std()
skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
print("sd:",sd/10000,"mean:",mean/10000,"mode:",mode/10000,"Q25",Q25/10000,"Q75",Q75/10000,"IQR",IQR/10000,"skew",skew,"kurt",kurt)
cent = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
print(cent)
#print(cent)
#entropy= spectral_entropy(signal, sf=fs_rate)
#print(entropy)
