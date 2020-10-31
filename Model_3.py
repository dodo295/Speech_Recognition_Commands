#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:07:55 2020

@author: Doaa
"""

#%% ================================ Importing Libraries ======================

from keras.layers import (BatchNormalization, Dense, Dropout, Flatten)
import numpy as np
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#%% ===================== Load variables from a .pickle file ==================  
N = 30   # Number of categories
with open('train_mfccs.pickle', 'rb') as f:
    X_train, y_train = pickle.load(f)  
    
with open('validation_mfccs.pickle', 'rb') as f:
    X_validation, y_validation = pickle.load(f)  
    
with open('test_mfccs.pickle', 'rb') as f:
    X_test, y_test = pickle.load(f)  
 
#%% ========================== Part 3 - Data Preprocessing ====================

# Creating one hot encoder object with categorical feature 0
def onehot_encoded(label,N):
    # N is the Number of categories
    onehot_encoded = list()
    for value in label:
        encoder = [0 for _ in range(N)]
        encoder[value] = 1
        onehot_encoded.append(encoder)
    return onehot_encoded

N = 30   # Number of categories    
y_train = onehot_encoded(y_train,N)
y_test = onehot_encoded(y_test,N)
y_validation = onehot_encoded(y_validation,N)

# Converting Dataset from list to Array
X_train  = np.array(X_train)
X_test  = np.array(X_test)
X_validation  = np.array(X_validation)
y_train  = np.array(y_train)
y_test  = np.array(y_test)
y_validation  = np.array(y_validation)

X_train, y_train = shuffle(X_train, y_train)


#%% ====================== Part 4 - Building the model ========================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%% Building the first model %%%%%%%%%%%%%%%%%%%%%%%%

from keras.models import Sequential
model_3 = Sequential()
model_3.add(Flatten(input_shape=(20, 44)))
model_3.add(BatchNormalization())
model_3.add(Dropout(rate = 0.1))
model_3.add(Dense(1024, activation = 'relu'))
model_3.add(Dropout(rate = 0.1))
model_3.add(BatchNormalization())
model_3.add(Dropout(rate = 0.1))
model_3.add(Dense(512, activation = 'relu'))
model_3.add(Dropout(rate = 0.1))
model_3.add(BatchNormalization())
model_3.add(Dropout(rate = 0.1))
model_3.add(Dense(N, activation='softmax'))
model_3.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

model_3.summary()

#%%%%%%%%%%%%%%%%%%%%%%%%%% Model Training & Testing %%%%%%%%%%%%%%%%%%%%%%%%%%

history_3 = model_3.fit(X_train, y_train, 
                batch_size=32, 
                epochs=50, 
                validation_data=(X_validation, y_validation), 
                verbose=1) 

model_3.evaluate(X_test, y_test) 

d_3= model_3.predict(X_test)

#model_3.save('model_3_9571_8648last.h5')


# loss
plt.plot(history_3.history['loss'], label='train loss')
plt.plot(history_3.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(history_3.history['accuracy'], label='train acc')
plt.plot(history_3.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
