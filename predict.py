#from Data_training import lab_enc
from cleaning_data import downsample, envelope
from tensorflow.keras.models import load_model
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
import librosa


def predict():

    model = load_model('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/models/AlexNet.h5')
    
    wav_paths = glob('{}/**'.format('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/wave_files/test'), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    
    classes = sorted(os.listdir('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/wave_files/test'))
    print('sha')
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    print('nand')
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []
##
##    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
##        X = np.empty((60, 40, 969), dtype=np.float32)
####        y, sr = librosa.load('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/wave_files/water/water_1.wav', sr=16000, mono=True)
##        y, sr = librosa.load(wav_fn, sr=16000, mono=True)                
##        mfccs = librosa.feature.mfcc(y=y, sr=16000, n_mfcc = 40)        
##        mfccsscaled = np.mean(mfccs.T,axis=0)
##
##        X[z,] = mfccsscaled.reshape(-1, 1)
##        print(X[z,])
##        print(X.shape)    
##        X[z,] = X.reshape(60, 40, 969, 1)
##        
##        predicted_vector = model.predict_classes(X)
##        predicted_class = le.inverse_transform(predicted_vector)
##        print("Predicted class",predicted_class[0])
##        predicted_proba_vector = model.predict_proba([X])
##
##
##        predicted_proba = predicted_proba_vector[0]
##
##        for i in range(len(predicted_proba)):
##            category = le.inverse_transform(np.array([i]))
##            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )





    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):

        y, sr = librosa.load(wav_fn, sr=16000, mono=True)
        
        mfccs = librosa.feature.mfcc(y=y, sr=16000, n_mfcc = 40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        step = int(16000)
        batch = []
        y_prob = []
        print(mfccsscaled.shape[0])

        
##        y, sr = librosa.load('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/wave_files/acc_audio/water/dc2e51fb-d281-464d-aa99-97d28b47f862_0.wav', sr=16000, mono=True)
        X = np.empty((18, 40, 969), dtype=np.float32)              
        mfccs = librosa.feature.mfcc(y=y, sr=16000, n_mfcc = 40)        
        mfccsscaled = np.mean(mfccs.T,axis=0)

        X[z,] = mfccsscaled.reshape(-1, 1)
        
        #print(X[z,])    
        X = X.reshape(18, 40, 969, 1)
##        print(z)    
        predicted_vector = model.predict_classes(X)
##        print(predicted_vector)
##        y_prob.append(predicted_vector)
##        y_mean = np.mean(predicted_vector,axis=0).flatten()
##        predicted_vector = np.argmax(y_mean)
##        print(predicted_vector)
##        results.append(y_mean)
        predicted_class = le.inverse_transform(predicted_vector)
        print("Predicted class",predicted_class[z])
        predicted_proba_vector = model.predict_proba([X])


        predicted_proba = predicted_proba_vector[z]
##
##        print(len(predicted_proba))
##        for i in range(len(predicted_proba)):
##            category = le.inverse_transform(np.array([i]))
##            print(category[z], "\t\t : ", format(predicted_proba[i], '.32f') )    












predict()
