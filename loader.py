#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 22:17:02 2020

@author: Doaa
"""
#%% ============================= Importing Libraries =========================
import os
import librosa
import glob
import numpy as np
import pickle

#%% ===================== Part 1 - Working with audio data ====================
"""The dataset we are using is Google's Speech Dataset 
(https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html).
It is composed of "65,000 one-second long utterances of 30 short words, 
by thousands of different people"."""

root = 'speech_commands_v0.01'
validation_file = 'validation_list.txt'
test_file = 'testing_list.txt'

classes = ["yes", "no", "up", "down", "left","right", "on", "off", "stop", "go", 
           "zero", "one", "two", "three", "four","five", "six", "seven", "eight",
           "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", 
           "sheila","tree", "wow"]
classes.sort()
class_to_idx = { classes[i]: i for i in range(len(classes)) }

#%% ================= Part 2 - Creating a custom audio Dataset ================
#You may have noticed that in this dataset, the test and validation datasets
#are given in testing_list.txt and validation_list.txt files.
#With that, we can infer a training list as well:

validation_filenames, test_filenames, train_filenames = [],[],[]

# Read the validation list
with open(os.path.join(root, validation_file)) as f:
    validation_filenames = [os.path.join(root, i) for i in f.read().strip().split('\n')]

# Read the test list    
with open(os.path.join(root, test_file)) as f:
    test_filenames = [os.path.join(root, i) for i in f.read().strip().split('\n')]
    
# Construct a train list    
for c in classes:
    train_filenames.extend([i for i in glob.glob(os.path.join(root, c, '*.wav'))
            if i not in validation_filenames and i not in test_filenames])

# print Number of samples for each list  
print('Training set: ',len(train_filenames))
print('Validation set : ',len(validation_filenames))
print('Test set: ',len(test_filenames))


#%% =============== Function for Createing a custom SpeechDataset class========
# ====================== that takes a file list in input.======================

def SpeechDataset(filename,saving_name):
    data_mfcc, data_label= [], []
    for i in range(len(filename)):
        data, sr = librosa.load(filename[i])
      # if the waveform is too short (less than 1 second) we pad it with zeroes
        if len(data) < 22050:
            data = np.pad(data, (0, 22050 - len(data)), 'constant')
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
        label = class_to_idx[filename[i].split('/')[1]]
        data_mfcc.append(mfccs)
        data_label.append(label)
	
    with open(saving_name+'.pickle', 'wb') as f:
        pickle.dump([data_mfcc, data_label], f)   
       
       
SpeechDataset(train_filenames,'train')
SpeechDataset(validation_filenames,'validation')
SpeechDataset(test_filenames,'test')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


