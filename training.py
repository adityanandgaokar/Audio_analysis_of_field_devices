import argparse
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from scipy.io import wavfile
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import pandas as pd
#from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from kapre.time_frequency import Melspectrogram
import librosa
from kapre.utils import Normalization2D
from tensorflow.keras.models import Model
from keras import regularizers
from tensorflow.keras.models import load_model



class DataGen(tf.keras.utils.Sequence):
    def __init__(self, wave_direc, labels, down_sample, sample_time,
                  no_classes, batch_size= 16, shuffle = True):
        self.wave_direc = wave_direc
        self.labels = labels
        self.down_sample = down_sample
        self.sample_time = sample_time
        self.no_classes = no_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()
        

    def __len__(self):
        # 'Denotes number of batches per epoch'
        return int(np.floor(len(self.wave_direc)/ self.batch_size))
            
    def __getitem__(self, index):
        # 'Generate one batch of data'
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        wave_direc = [self.wave_direc[e] for e in indexes]
            
        labels = [self.labels[e] for e in indexes]
        
        X = np.empty((self.batch_size, 40, 969), dtype=np.float32)
        Y = np.empty((self.batch_size, self.no_classes), dtype=np.float32)


        for i, (path, label) in enumerate(zip(wave_direc, labels)):
            y, sr = librosa.load(path, sr=16000, mono=True)
            mfccs = librosa.feature.mfcc(y=y, sr=self.down_sample, n_mfcc = 40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            #rate, wav = wavfile.read(path)
            #print(wav.shape[0])
            #print(wav)
            X[i,] = mfccsscaled.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.no_classes)
            
        

        X = X.reshape(self.batch_size, 40,969,1)
        
        return X, Y         
                
    
    def on_epoch_end(self):
        # updating indexes after each epoch
        
        self.indexes = np.arange(len(self.wave_direc))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            

            

def CONVOLUTION2D(no_classes, sample_rate, sample_time, X_train,Y_train, Y_test, X_test, batch_size=2):

    model_selection = args.model_selection
    csv_path = os.path.join('adityanandgaokar/Sound_classification','{}_history.csv'.format(model_selection))


      
    if K.image_data_format() == "channels_first":
        
        input_shape = (1, 40, 969)
        channeldim = 1
       
    else:
        input_shape = (40,969,1)
        channeldim = -1
        
    model = Sequential()
    model.add(Conv2D(96, 11, strides= 4, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(3,2, padding='same'))
    model.add(Conv2D(256,5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(3,2, padding='same'))
    model.add(Conv2D(384, 3, strides= 1, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, strides= 1, padding='same', activation='relu'))
    model.add(MaxPooling2D(3,2, padding='same'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))

    
"""you could also use convolution neural network 2d which has written below one""" 
"""Only you have to change input variable i"""
    
##    i = layers.Input(shape=(40, 32), name='input')
##    
##    y = Melspectrogram(n_dft=32, n_hop=10,
##                       padding='same', sr=16000, n_mels=8,
##                       fmin=0.0, fmax=16000/2, power_melgram=1.0,
##                       return_decibel_melgram=True, trainable_fb=False,
##                       trainable_kernel=False,
##                       name='melbands')(i)
##     
##    x = Normalization2D(str_axis='batch', name='batch_norm')(y)
##    x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
##    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
##    #x = layers.Dropout(rate=0.2)(x)
##    x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
##    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
##    #x = layers.Dropout(rate=0.2)(x)
##    x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
##    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
##    #x = layers.Dropout(rate=0.2)(x)
##    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
##    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
##    #x = layers.Dropout(rate=0.2)(x)
##    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
##    x = layers.Flatten(name='flatten')(x)
##    x = layers.Dropout(rate=0.5)(x)
##    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.0001), name='dense')(x)
##    o = layers.Dense(no_classes, activation='softmax', name='softmax')(x)
##    
##    model = Model(inputs=i, outputs=o, name='2d_convolution')
    
    
    
"""you could also use strided convolution neural network you just have give right input variable according to your data"""


##    y = Normalization2D(str_axis='batch', name='batch_norm')(y)
##    y = layers.Conv2D(16, kernel_size=(7,7), strides= (2,2), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='CONVOLUTION2D_16')(y)
##    
##    
##    
##    y = layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='CONVOLUTION2D_relu1')(y)
##    y = layers.Activation('relu')(y)
##    y = layers.BatchNormalization(axis=channeldim)(y)
##    y = layers.Conv2D(32, kernel_size=(3,3), strides= (2,2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='CONVOLUTION2D_relu2')(y)
##    y = layers.Activation('relu')(y)
##    y = layers.BatchNormalization(axis=channeldim)(y)
##    y = layers.Dropout(0.1)(y)        
##
##    y = layers.Conv2D(64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='CONVOLUTION2D_relu3')(y)
##    y = layers.Activation('relu')(y)
##    y = layers.BatchNormalization(axis=channeldim)(y)
##    y = layers.Conv2D(64, kernel_size=(3,3), strides= (2,2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='CONVOLUTION2D_relu4')(y)
##    y = layers.Activation('relu')(y)
##    y = layers.BatchNormalization(axis=channeldim)(y)
##    y = layers.Dropout(0.1)(y)
##    print('haha')
##
##    
##    y = layers.Conv2D(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='CONVOLUTION2D_relu5')(y)
##    y = layers.Activation('relu')(y)
##    y = layers.BatchNormalization(axis=channeldim)(y)
##    y = layers.Conv2D(128, kernel_size=(3,3), strides= (2,2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='CONVOLUTION2D_relu6')(y)
##    y = layers.Activation('relu')(y)
##    y = layers.BatchNormalization(axis=channeldim)(y)
##    y = layers.Dropout(0.1)(y)
##
##    z = layers.Flatten(name='flatten')(y)
##    z = layers.Dense(512, kernel_initializer='he_normal')(z)
##    z = layers.Activation('relu')(z)
##    z = layers.BatchNormalization()(z)
##    z = layers.Dropout(0.2)(z)    
##
##    z = layers.Dense(no_classes)(z)
##    a = layers.Activation('softmax')(z)
##    model = Model(inputs= i, outputs= a, name='conv2d')
    
    
    model.summary()
    #model = Model(inputs= x, outputs= z, name='conv2d')
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model_checkpoint_callback = ModelCheckpoint('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/models/{}.h5'.format(model_selection),
                                    monitor = 'val_acc', save_best_only = True,
                                    save_weights_only=False,mode = 'auto',
                                    save_freq='epoch', verbose=1)
    csv_log = CSVLogger(csv_path, append=False)

   

    print(training_data)

    model.fit(training_data, validation_data= validate_data, epochs = 30,
               verbose = 1, callbacks=[csv_log,model_checkpoint_callback])




    model.save('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/models/AlexNet.h5')#.h5'#.format(model_selection))



    
    return model


def predict(model_file , lab_enc, file_name):
    model = load_model(model_file, custom_objects={'Melspectrogram':Melspectrogram, 'Normalization2D':Normalization2D})
    wave, sr = librosa.load(file_name, sr=16000, mono=True)
            
    mfccs = librosa.feature.mfcc(y=wave, sr=16000, n_mfcc = 40)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    
    X1 = np.empty((16, 40, 969), dtype=np.float32)

    mfccsscaled = np.random.randint(0,2, 16*40*969)
    mfccsscaled = mfccsscaled.reshape(16, 40, 969)
    data = tf.cast(mfccsscaled, tf.float32)
    y_pred = model.predict(data)
    y_mean = np.mean(y_pred,axis=0).flatten()
    y_pred = np.argmax(y_mean)
    print(y_pred)
       
         
def Data_training(args):
    source_fls = args.source_fls
    batch_size = args.batch_size
    down_sample = args.down_sample
    sample_time = args.sample_time
    number_of_classes = len(os.listdir(args.source_fls))
    model_selection = args.model_selection
    parameters = {'no_classes': number_of_classes,
                  'sample_rate' : down_sample,
                  'sample_time' : sample_time, 
                  }
    
    csv_path = os.path.join('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/logs','{}_history.csv'.format(model_selection))
    wave_direc = glob('{}/**'.format(source_fls), recursive =True)
    class_list = sorted(os.listdir(args.source_fls))
    wave_direc = [x.replace(os.sep, '/') for x in wave_direc if '.wav' in x]
    lab_enc = LabelEncoder()
    lab_enc.fit(class_list)
    #labels = [os.path.split(x)[0].split('/')[-1] for x in wave_direc]
    labels = [os.path.dirname(x).split('/')[-1] for x in wave_direc]    
    labels = lab_enc.transform(labels)
        
    
    X_train, X_test, Y_train, Y_test = train_test_split(wave_direc, labels, test_size= 0.1, random_state = 0)

    
    
    global training_data
    




    training_data = DataGen(X_train, Y_train, down_sample, sample_time,
                            len(set(Y_train)), batch_size = batch_size)
    
    
    global validate_data

    
    validate_data = DataGen(X_test, Y_test, down_sample, sample_time,
                            len(set(Y_test)), batch_size = batch_size)




    params = {'X_train' : X_train,
              'Y_train' :Y_train,
              'Y_test' : Y_test,
              'X_test' : X_test,
              'batch_size' : batch_size}


    
    model = CONVOLUTION2D(no_classes = 2, sample_rate = 16000, sample_time = 1.0, **params)

        
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Data Training')
    parser.add_argument('--model_selection', type=str, default = 'convolution2d',
                         help = 'model which we want for training the data')
    parser.add_argument('--source_fls', type=str, default ='C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/wave_files/clean',
                         help = 'cleaned files as source files')
    parser.add_argument('--batch_size', type=int, default=16,
                         help = 'batch size')
    parser.add_argument('--down_sample', type=int, default=16000,
                         help = 'down sample rate')
    parser.add_argument('--sample_time', type=float, default=1.0,
                         help = 'audio samples in a seconds')
    
    
    args, _ = parser.parse_known_args()

    Data_training(args)
    
