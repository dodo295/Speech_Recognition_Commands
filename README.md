# Dataset
   • **Speech Commands Data Set v0.01**
      
      This dataset containing 64,727 audio files, released on August 3rd 2017. 
      This is a set of one-second .wav audio files,each containing a single spoken English word.
      These words are from a small set of commands, and are spoken by a variety 
      of different speakers. The audio files are organized into folders based on the word they contain, 
      and this data set is designed to help train simple machine learning models.

   • Its original location was at[Click here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

   • **Data Coverage**
      
      20 of the words are core words, while 10 words are auxiliary words that
      could act as tests for algorithms in ignoring speeches that do not contain triggers. 
      
    • Core words: Yes, No, Up, Down, Left, Right, On, Off, Stop, Go, Zero, One,
                  Two, Three, Four, Five, Six, Seven, Eight, and Nine.
      
    • Auxiliary words: Bed, Bird, Cat, Dog, Happy, House, Marvin, Sheila, Tree, and Wow.


   • **Data Split**
      
      Train – 51,088 audio clips, Validation – 6,798 audio clips, Test – 6,835 audio clips.

# Speech Feature Extraction
   • **MFCC**: Mel-frequency cepstral coefficients calculation
   
     MFCC feature extraction technology is less complicated to implement, more effective and robust
     under various conditions, and with the help of this technique we can normalize features as well,
     which is a very popular technique for recognizing isolated words in English language.
     Features are extracted based on the information that is included in the speech signal.
     
   • The MFCC approach is used in this project to extract features from the speech signal for spoken words.

      
    • loader.py : Speech pre-processing, including data loading, and very short 
                  waveform stuffing (less than one second),we pad it with zeros,
                  and compute the MFCC feature.
    • These features are saved in Pickle files for each training, validation, and testing dataset.
   You will find the test and validation files in the mfcc_feature directory and you can download
   the train pickle from this link:[Click here](https://drive.google.com/file/d/1WRHmXAPFMHimIaNdgyHBbVKS5xnTpC0d/view?usp=sharing)
   
# Description of Model Layers
• **Input**

    I use MFCC as the model’s input features. It consists of 20-dimensional MFCC.
• **Output**

    The output phoneme classes is reduced to 30 classes during evaluation.

• **Batchnormalization Layer**

    • Training of deep neural networks is complicated by the fact that the input distribution
      of each layer changes during training, as the parameters of the previous layers change. 
    
    • This change was defined as an endogenous shift. Batchnormalization is designed to mitigate
      this internal covariate transformation by introducing a normalization step that ﬁxes the
      means and variances of layer inputs.
    
    • Batch Normalization (BN) is widely used in deep learning and is significantly improving
      in many tasks. It helps accelerate training speed and greatly improve performance.
      In this work,I use BN sometimes after each layer and sometimes only at the beginning.
      
 • **Clipped ReLU Activation**
 
    The Clipped ReLU is a review of ReLU. It presents a coefficient of α> 0. Its product for each
    element greater than α is α. Hence, Clipped ReLU limits the output to {0, α}.

# Models
• **First Model (Model_1.py)**
    
    #Layer 1 BatchNormalization with input shape of mfcc features
    model_1.add(BatchNormalization(input_shape = (20,44)))
    
    #Layer 2 with clipped ReLu activation function
    model_1.add(Dense(512, activation = clipped_relu, input_shape=(20,44)))
    model_1.add(Dropout(rate = 0.1))
    
    #Layer 3 with clipped ReLu activation function
    model_1.add(Dense(256, activation = clipped_relu))
    model_1.add(Dropout(rate = 0.1))
    
    #Layer 4 Recurrent layer with clipped ReLu activtion function
    model_1.add(SimpleRNN(512, activation = clipped_relu, return_sequences = True))
    model_1.add(Dropout(rate = 0.1))
    
    #Layer 5 Flatten
    model_1.add(Flatten())
    
    #Layer 6 with softmax activaiton function and units equal umber of classes
    model_1.add(Dense(units = 30, activation = "softmax"))
    
• **Second Model (Model_2.py)**
    
    # Layer 1 BatchNormalization layer with 1D mfcc feature
    model_2.add(BatchNormalization(input_shape = (880,)))
    
    # Layer 2 with clipped Relu activaiton function
    model_2.add(Dense(256,input_shape=(880,),activation = clipped_relu)) 
    model_2.add(Dropout(0.1)) 
    
    # Layer 3 with clipped Relu activaiton function
    model_2.add(Dense(256, activation = clipped_relu)) 
    model_2.add(Dropout(0.1)) 
    
    # Layer 4 with softmax activaiton function and units equal umber of classes
    model_2.add(Dense(N)) 
    model_2.add(Activation('softmax')) 

• **Third Model (Model_3.py)**
    
    # Layer 1 Flatten layer with 2D mfcc feature
    model_3.add(Flatten(input_shape=(20, 44)))
    
    # Layer 2 BatchNormalization layer
    model_3.add(BatchNormalization())
    model_3.add(Dropout(rate = 0.1))
    
    # Layer 3 with Relu activaiton function
    model_3.add(Dense(1024, activation = 'relu'))
    model_3.add(Dropout(rate = 0.1))
    
    # Layer 4 BatchNormalization layer
    model_3.add(BatchNormalization())
    model_3.add(Dropout(rate = 0.1))
    
    # Layer 5 with Relu activaiton function
    model_3.add(Dense(512, activation = 'relu'))
    model_3.add(Dropout(rate = 0.1))
    
    # Layer 6 BatchNormalization layer
    model_3.add(BatchNormalization())
    model_3.add(Dropout(rate = 0.1))
    
    # Layer 7 with softmax activaiton function and units equal umber of classes
    model_3.add(Dense(N, activation='softmax'))

**These models are saved in the Templates folder as an h5 file.**
   Model   |  Training Accuracy | Testing Accuracy 
   --------|--------------------|--------------------------------
   Model_1 | 88.8%   | 82.96%
   Model_2   | 92.05% |  83.57%
   Model_3   | 97.76%  |   86.63%

# Real-time Speech Recognition Commands

    I used the three speech recognition models to recognize the speech command performed 
    when a human speaks within just one second once the microphone starts streaming 
    with the pyaudio python library.
    So we obtain the expected resulting speech command by repeated prediction from 3 models.
You will find it in main.py file.

