#------Import and Install Dependencies--------------
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
#----------------------------------------------------

#check the current working directory
pwd

#--------Build Data Loading Function-----------------
#Files used to test data loading function
#Define file paths
BLRA_FILE = 'C:\\Users\\chris\\Desktop\\Black_Rail_Audio_Detection\\data\\interim\\Parsed_BLRA_Clips\\BR_Call_1.wav'

NOT_BLRA_FILE = 'C:\\Users\\chris\\Desktop\\Black_Rail_Audio_Detection\\data\\interim\\Parsed_Not_BLRA_Clips\\afternoon-birds-song-in-forest-1.wav'

#create function that resamples audio to 16Hz
#create tensor so that machine learning can be done
def load_wav_16k_mono(file_name):
    #Load encoded wav file
    file_contents = tf.io.read_file(file_name)
    # Decode wav (tensor by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

#Let's see if our function worked
#Plot Wave
wave = load_wav_16k_mono(BLRA_FILE)
nwave = load_wav_16k_mono(NOT_BLRA_FILE)

plt.plot(wave) # blue wave
plt.plot(nwave) # orange wave
plt.show()
#-------------------------------------------------------------------------

#-----Create Tensorflow Dataset-------------------------------------------
#Create Data Pipeline
#Define wich folders contain the clips that have BLRA calls and which do not
POS = 'C:\\Users\\chris\\Desktop\\Black_Rail_Audio_Detection\\data\\interim\\Parsed_BLRA_Clips'
NEG = 'C:\\Users\\chris\\Desktop\\Black_Rail_Audio_Detection\\data\\interim\\Parsed_Not_BLRA_Clips'

# Create a variable that holds all of the files in each folder - searches for files with .wav at the end
pos = tf.data.Dataset.list_files(POS+'\*.wav')
neg = tf.data.Dataset.list_files(NEG+'\*.wav')

# Make sure the previous code worked - should see any .wav file within the positive folder
pos.as_numpy_iterator().next()

# How many .wav files are in the BLRA positive folder?
len(pos)
# How many .wav files are in the not BLRA folder?
len(neg)

# Add labels to each file and combine positive and negative samples into variable called "data"
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives) #Added label (either 1 or 0) to each file to represent pos or neg call
len(data) # Check and see if data includes both positives and negatives
#----------------------------------------------------------------------------------------------------

#------Determine the average length of a black rail call---------------------------------------------
# Calculate wave cycle length
lengths=[]
for file in os.listdir(POS):
    tensor_wave = load_wav_16k_mono(os.path.join(POS, file))
    lengths.append(len(tensor_wave))
    
# Show the array we just created of call lengths
lengths

# Note: There are several repetitions because the 210 clips were produced by copying and pasting the same 30 
# .wav files within the positives folder

# Moving forward, we need to specify call length, so let's get some summary statistics
# Calculate Mean, Min, and Max
mean = tf.math.reduce_mean(lengths)
min = tf.math.reduce_min(lengths)
max = tf.math.reduce_max(lengths)

print(mean, min, max)
#--------------------------------------------------------------------------------------------


#-----Convert Audio File to Spectrogram------------------------------------------------------

# Build preprocessing function
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path) # Our data loading function
    wav = wav[:15000] # average length of BLRA call 
    zero_padding = tf.zeros([15000] - tf.shape(wav), dtype=tf.float32) # make sure we include all files below 15000
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# Test out the function and viz of the spectrogram from the positives files
# Create a spectrogram from a black rail call
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)

# Plot the spectrogram
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()
#---------------------------------------------------------------------------------------------------------------------



#-------Create training and testing partitions------------------------------------------------------------------------

# Create a tensorflow data pipeline
data = data.map(preprocess) #converts all audio files to spectrograms
data = data.cache() 
data = data.shuffle(buffer_size=1000) # shuffle between pos & neg files - reduce bias
data = data.batch(16) # Train on 16 samples at a time
data = data.prefetch(8)

# What does our data look after preprocessing?
# Remember we are running test in batches of 16 
# so len(data) will return no. approx. 16X lower than total number of pos and neg files
len(data)

# Split into training and testing data 
# based on results of the previous code (i.e., take ~70% of data for training partition)
train = data.take(19)
test = data.skip(19).take(8)

# Test One Batch from the training data
# First number should be 16 - number of files in the batch
# Last 3 numbers are shape of spectrogram - need to pass through deep learning model
samples, labels = train.as_numpy_iterator().next()
samples.shape

# Check that the batch has mix of positive (1) and negative (1) clips
labels

#-------Build Deep Learning Model---------------------------------------------------------
# Load Tensorflow Dependencies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Build sequential model, compile and view summary
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(459, 257,1))) #numbers taken from shape of spectorgram *samples.shape*
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
# Check out the parameters
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.summary()
#---------------------------------------------------------------------------------------------------------------------------


#-----------Train the Model-------------------------------------------------------------------------------------------------
# Fit Model, View Loss and KPI Plots
hist = model.fit(train, epochs=4, validation_data=test)

# Get some numbers of how the model performed
hist.history

# Plot loss
plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()
#---------------------------------------------------------------------------------------------------------------------------


#-------Make a prediction on a single clip------------------------------------------------------------------
# Get one batch and make a prediction
X_test, y_test = test.as_numpy_iterator().next()

# Make sure the test includes 16 spectrograms
X_test.shape

# Make sure the y_test has 16 labels
y_test.shape

# Use your model to make some predictions!
yhat = model.predict(X_test)

# See how your model did - get logits or confidence metrics for each spectrogram in the batch
yhat

# Convert Logits to Classes
# Loop through each prediction - model is precise so raise confidence interval - let's say 95%
yhat = [1 if prediction > 0.95 else 0 for prediction in yhat]

# Display the new classes 
yhat

# How many rails were heard based on the model prediction?
tf.math.reduce_sum(yhat)

# How many rails were heard based on the test data?
tf.math.reduce_sum(y_test)

# Look at y_test as classes
y_test.astype(int)

# Try out the model on some new audio files
#Build parsing functions
#Load up MP3s

def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

mp3 = 'C:\\Users\\Soldo\\Desktop\\Black_Rail_Audio_Files\\BLRA_Deep_Learning_Model\\Audio_Files_To_Test_Model\\BLRA_Test_Clip2_Pos.mp3'
mp3

wav = load_mp3_16k_mono(mp3)

# Convert "wav" file into slices for image recognition
# Remember - our sequence length is 15,000
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=15000, sequence_stride=15000, batch_size=1)

# Take one clip and slice it up
samples, index = audio_slices.as_numpy_iterator().next()
len(audio_slices)

#Build function to convert clip slices into windowed spectrograms
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([15000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# Convert longer clips into windows and make predicitons
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=15000, sequence_stride=15000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)

# Make predicitons on the mp3 files
yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.95 else 0 for prediction in yhat]

# Take a look at our predictions
yhat

len(yhat)

# Group consecutive detections
from itertools import groupby

yhat = [key for key, group in groupby(yhat)]
tf.math.reduce_sum(yhat)
calls = tf.math.reduce_sum(yhat).numpy()
calls

#Make Prediction
#Loop over all recordings and make predictions

results = {}
for file in 'C:\\Users\\Soldo\\Desktop\\Black_Rail_Audio_Files\\BLRA_Deep_Learning_Model\\Audio_Files_To_Test_Model':
    FILEPATH = ('C:\\Users\\Soldo\\Desktop\\Black_Rail_Audio_Files\\BLRA_Deep_Learning_Model\\Audio_Files_To_Test_Model', file)
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=15000, sequence_stride=15000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    results[file] = yhat
    
#Convert Predictions into Classes
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
class_preds

# Group Consecutive Detections
postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessed

# Export Results
import csv

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])        