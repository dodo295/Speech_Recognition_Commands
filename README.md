# Dataset
   • Speech Commands Data Set v0.01
      
      This dataset containing 64,727 audio files, released on August 3rd 2017. This is a set of one-second .wav audio files,
      each containing a single spoken English word. These words are from a small set of commands, and are spoken by a variety 
      of different speakers. The audio files are organized into folders based on the word they contain, 
      and this data set is designed to help train simple machine learning models.

    • Its original location was at http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz.

   • Data Coverage
      
      20 of the words are core words, while 10 words are auxiliary words that could act as tests for algorithms in ignoring speeches that do not contain triggers. 
      
    • Core words: Yes, No, Up, Down, Left, Right, On, Off, Stop, Go, Zero, One, Two, Three, Four, Five, Six, Seven, Eight, and Nine.
      
    • Auxiliary words: Bed, Bird, Cat, Dog, Happy, House, Marvin, Sheila, Tree, and Wow.


   • Data Split
      
      Train – 51,088 audio clips, Validation – 6,798 audio clips, Test – 6,835 audio clips


# Speech Feature Extraction
   • MFCC: Mel-frequency cepstral coefficients calculation
      
    • loader.py : Speech pre-processing, including data loading, and very short waveform stuffing (less than one second), we pad it with zeros, and compute the MFCC feature.
    • These features are saved in Pickle files for each training, validation, and testing dataset. You will find it in the mfcc_feature directory.
# Input & Output
The input feature is MFCC 20 , and the output phoneme classes is reduced to 30 classes during evaluation.
