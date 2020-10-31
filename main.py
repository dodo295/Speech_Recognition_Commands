#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:52:55 2020

@author: Doaa
"""

import pyaudio
import wave
import librosa
import numpy as np
from keras.models import load_model
from keras.activations import relu
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

classes = ["yes", "no", "up", "down", "left","right", "on", "off", "stop", "go", 
           "zero", "one", "two", "three", "four","five", "six", "seven", "eight",
           "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", 
           "sheila","tree", "wow"]
classes.sort()
class_to_idx = { classes[i]: i for i in range(len(classes)) }


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

input_data = []
data, sr = librosa.load(WAVE_OUTPUT_FILENAME)
# if the waveform is too short (less than 1 second) we pad it with zeroes
if len(data) < 22050:
    data = np.pad(data, (0, 22050 - len(data)), 'constant')
mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
input_data.append(mfccs)   
input_data = np.array(input_data)

feature_dim_1 = input_data.shape[1]
feature_dim_2 = input_data.shape[2]

input_data2 = input_data.reshape((1,feature_dim_1*feature_dim_2))

def clipped_relu(x):
    return relu(x, max_value=20)

get_custom_objects().update({"clipped_relu": clipped_relu})
K.set_learning_phase(1) 

model_1 = load_model('model_1_8880_8296.h5')
model_2 = load_model('model_2_9205_835687.h5')
model_3 = load_model('model_3_9776_8663.h5')

predicted = np.concatenate(
        (model_1.predict(input_data),
        model_2.predict(input_data2),
        model_3.predict(input_data))
                           )
   
max_value = np.amax(predicted, axis = 1)
max_index = np.argmax(predicted, axis = 1)
predicted_class = np.argmax(np.bincount(max_index))

for word, idx in class_to_idx.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if idx == predicted_class:
        predicted_word = word
        print(word)
