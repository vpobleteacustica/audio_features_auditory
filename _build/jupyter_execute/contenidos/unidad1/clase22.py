#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We require to install the following packages:
# !pip install torchaudio librosa

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)


# # Paisaje sonoro: Bar muy ruidoso y separación de fuentes
# 
# > https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html

# In[2]:


import os
import librosa
import matplotlib.pyplot as plt
import requests
from IPython.display import Audio, display


# In[3]:


def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_speech_sample(audio_path='', resample=None):
    return _get_sample(audio_path, resample=resample)

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)

def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


# In[4]:


_SAMPLE_DIR = "bar"
audio_path_1 = os.path.join(_SAMPLE_DIR, "bar.wav")
audio_path_2 = os.path.join(_SAMPLE_DIR, "bass.wav")
audio_path_3 = os.path.join(_SAMPLE_DIR, "cheering.wav")
audio_path_4 = os.path.join(_SAMPLE_DIR, "drums.wav")
audio_path_5 = os.path.join(_SAMPLE_DIR, "guitar.wav")
audio_path_6 = os.path.join(_SAMPLE_DIR, "voice.wav")


os.makedirs(_SAMPLE_DIR, exist_ok=True)


# In[5]:


waveform1, sample_rate = get_speech_sample(audio_path_1)
waveform2, sample_rate = get_speech_sample(audio_path_2)
waveform3, sample_rate = get_speech_sample(audio_path_3)
waveform4, sample_rate = get_speech_sample(audio_path_4)
waveform5, sample_rate = get_speech_sample(audio_path_5)
waveform6, sample_rate = get_speech_sample(audio_path_6)

t = 10

muestra1 = waveform1[:,1:44100*t]
plot_waveform(muestra1, sample_rate, title="bar")
play_audio(muestra1, sample_rate)


# In[6]:


muestra2 = waveform2[:,1:44100*t]
plot_waveform(muestra2, sample_rate, title="bass")
play_audio(muestra2, sample_rate)


# In[7]:


muestra3 = waveform3[:,1:44100*t]
plot_waveform(muestra3, sample_rate, title="cheering")
play_audio(muestra3, sample_rate)


# In[8]:


muestra4 = waveform4[:,1:44100*t]
plot_waveform(muestra4, sample_rate, title="drums")
play_audio(muestra4, sample_rate)


# In[9]:


muestra5 = waveform5[:,1:44100*t]
plot_waveform(muestra5, sample_rate, title="guitar")
play_audio(muestra5, sample_rate)


# In[10]:


muestra6 = waveform6[:,1:44100*t]
plot_waveform(muestra6, sample_rate, title="voice")
play_audio(muestra6, sample_rate)


# In[11]:


muestra1 = waveform1[:,1:44100*t]
muestra2 = waveform2[:,1:44100*t]
muestra3 = waveform3[:,1:44100*t]
muestra4 = waveform4[:,1:44100*t]
muestra5 = waveform5[:,1:44100*t]
muestra6 = waveform6[:,1:44100*t]

sum      = 5*muestra1 + 20*muestra2 + 5*muestra3 + 5*muestra4 + 5*muestra5 + 5*muestra6
plot_waveform(sum, sample_rate, title="soundscape")
play_audio(sum, sample_rate)
torchaudio.save('suma.wav', sum, sample_rate)

