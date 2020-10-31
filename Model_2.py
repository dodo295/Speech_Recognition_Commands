#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:37:18 2020

@author: Doaa
"""
#%% ================================ Importing Libraries ======================

import numpy as np
import pickle
import matplotlib.pyplot as plt
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras import backend as K
from keras.activations import relu
from keras.utils.generic_utils import get_custom_objects
from sklearn.utils import shuffle

#%% ===================== Load variables from a .pickle file ==================  
N = 30   # Number of categories
librosa_sample_rate = 22050
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

# Extracting the 2 dimentions of MFCCs features
feature_dim_1 = X_train.shape[1]
feature_dim_2 = X_train.shape[2]
X_train_records  = X_train.shape[0]
X_test_records = X_test.shape[0]
X_validation_records = X_validation.shape[0]


X_train = X_train.reshape((X_train_records,feature_dim_1*feature_dim_2))
X_test = X_test.reshape((X_test_records,feature_dim_1*feature_dim_2))
X_validation = X_validation.reshape((X_validation_records,feature_dim_1*feature_dim_2))

#%% ====================== Part 4 - Building the model ========================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%% Building the first model %%%%%%%%%%%%%%%%%%%%%%%%

# Clipped ReLu function 
def clipped_relu(x):
    return relu(x, max_value=20)

get_custom_objects().update({"clipped_relu": clipped_relu})
K.set_learning_phase(1) 



model_2 = Sequential() 
# Layer 1 BatchNormalization layer
model_2.add(BatchNormalization(input_shape = (880,)))

# Layer 2 with Relu activaiton function
model_2.add(Dense(256,input_shape=(880,),activation = clipped_relu)) 
#model_2.add(Activation('relu')) 
model_2.add(Dropout(0.1)) 


# Layer 4 with Relu activaiton function
model_2.add(Dense(256, activation = clipped_relu)) 
#model_2.add(Activation('relu')) 
model_2.add(Dropout(0.1)) 

# Layer 3 Flatten
#model_2.add(Flatten())

# Layer 6 with softmax activaiton function
model_2.add(Dense(N)) 
model_2.add(Activation('softmax')) 

# Compile the model 
model_2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 

# Display model architecture summary 
model_2.summary() 

#%%%%%%%%%%%%%%%%%%%%%%%%%% Model Training & Testing %%%%%%%%%%%%%%%%%%%%%%%%%%

# Calculate pre-training accuracy 
num_epochs = 100
num_batch_size = 32 
history_2 = model_2.fit(X_train, y_train, 
                        batch_size=num_batch_size,
                        epochs=num_epochs, 
                        validation_data=(X_validation, y_validation), 
                        verbose=1) 
#model_2.save('model_2_9205_835687.h5')

# Evaluateing and Calculating the Pre-training Accuracy of the model
score = model_2.evaluate(X_test, y_test, verbose=0) 
accuracy = 100*score[1] 
print("Pre-training accuracy: %.4f%%" % accuracy)

# Calculating the Prediction of the model on Test Data
d_2 = model_2.predict(X_test)

# loss
plt.plot(history_2.history['loss'], label='train loss')
plt.plot(history_2.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(history_2.history['accuracy'], label='train acc')
plt.plot(history_2.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
