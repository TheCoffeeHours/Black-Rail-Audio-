#------Black Rail Audio Detection Pipeline------------------------------------------------------
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

def detect_black_rail_calls(mp3_path, model):
    def load_mp3_16k_mono(filename):
        res = tfio.audio.AudioIOTensor(filename)
        tensor = res.to_tensor()
        tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
        sample_rate = res.rate
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
        return wav

    def preprocess_mp3(sample, index):
        sample = sample[0]
        zero_padding = tf.zeros([15000] - tf.shape(sample), dtype=tf.float32)
        wav = tf.concat([zero_padding, sample],0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram

    wav = load_mp3_16k_mono(mp3_path)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=15000, sequence_stride=15000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    
    total_frames = len(list(audio_slices.as_numpy_iterator()))
    sampled_indices = np.random.choice(total_frames, total_frames // 2, replace=False)
    audio_slices_list = [list(audio_slices.as_numpy_iterator())[i] for i in sampled_indices]
    audio_slices_sampled = tf.data.Dataset.from_tensor_slices(audio_slices_list)
    audio_slices_sampled = audio_slices_sampled.batch(64)
    
    yhat = model.predict(audio_slices_sampled)
    yhat_binary = [1 if prediction > 0.95 else 0 for prediction in yhat]
    
    window_length_in_seconds = 1
    positive_windows_start_times = [i * window_length_in_seconds for i, prediction in enumerate(yhat_binary) if prediction == 1]
    
    return positive_windows_start_times

# Usage:
model = tf.keras.models.load_model('c:\\Users\\chris\\Desktop\\Black_Rail_Audio_Detection\\models\\BLRA_model.h5')
mp3_path = 'c:\\Users\\chris\\Desktop\\Black_Rail_Audio_Detection\\data\\Audio_Files_To_Test_Model\\BLRA_Test_Clip1_Pos.mp3'
positive_times = detect_black_rail_calls(mp3_path, model)
print("Positive predictions were made at the following start times (in seconds):", positive_times)
