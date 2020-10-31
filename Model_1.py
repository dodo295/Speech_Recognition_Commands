#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 00:57:07 2020

@author: Doaa
"""
#%% ================================ Importing Libraries ======================
import numpy as np
import pickle
import matplotlib.pyplot as plt
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, SimpleRNN, Flatten
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

#%% ====================== Part 4 - Building the model ========================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%% Building the first model %%%%%%%%%%%%%%%%%%%%%%%%

# Clipped ReLu function 
def clipped_relu(x):
    return relu(x, max_value=20)

get_custom_objects().update({"clipped_relu": clipped_relu})
K.set_learning_phase(1) 


model_1 = Sequential()
model_1.add(BatchNormalization(input_shape = (20,44)))
# Layer 1 with clipped ReLu activation function
model_1.add(Dense(512, activation = clipped_relu, input_shape=(20,44)))
model_1.add(Dropout(rate = 0.1))


# Layer 2 with clipped ReLu activation function
model_1.add(Dense(256, activation = clipped_relu))
model_1.add(Dropout(rate = 0.1))


# Layer 4 Bidirectional Recurrent layer with clipped ReLu activtion function
model_1.add(SimpleRNN(512, activation = clipped_relu, return_sequences = True))
model_1.add(Dropout(rate = 0.1))

# Layer 6 Flatten
model_1.add(Flatten())

# Layer 6 with softmax activaiton function
model_1.add(Dense(units = 30, activation = "softmax"))

# Compiling the model
model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

# Display model architecture summary 
model_1.summary()

#%%%%%%%%%%%%%%%%%%%%%%%%%% Model Training & Testing %%%%%%%%%%%%%%%%%%%%%%%%%%
# Fitting & Saving the model
model_1.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_validation,y_validation))

#model_1.save('model_1_8949_8019.h5')

# Evaluateing the model 
history_1 = model_1.evaluate(X_test, y_test) 

# Calculating the Pre-training Accuracy of the model
accuracy = 100*history_1[1] 
print("Pre-training accuracy: %.4f%%" % accuracy)

# Calculating the Prediction of the model on Test Data
d_1 = model_1.predict(X_test)

# loss
plt.plot(history_1.history['loss'], label='train loss')
plt.plot(history_1.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(history_1.history['accuracy'], label='train acc')
plt.plot(history_1.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
