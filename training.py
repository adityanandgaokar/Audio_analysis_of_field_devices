import argparse
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Alexnet
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from scipy.io import wavfile
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import pandas as pd
import librosa



class DataGen(tf.keras.utils.Sequence):
    def __init__(self, wave_direc, labels, down_sample, sample_time,
                  no_classes, batch_size= 32, shuffle = True):
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
            
            X[i,] = mfccsscaled.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.no_classes)
            

        X = X.reshape(self.batch_size, 40,969,1)
        
        return X, Y
         
    
    def on_epoch_end(self):
        # updating indexes after each epoch
        
        self.indexes = np.arange(len(self.wave_direc))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            

            


       
         
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
    models = {'Alexnet': Alexnet(**parameters)}#,
              #'convolution2d': CONVOLUTION2D(**parameters)}
    assert model_selection in models.keys(), '{} not an available model'.format(model_selection)

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


    model = models[model_selection]

    model_checkpoint_callback = ModelCheckpoint('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/models/{}.h5'.format(model_selection),
                                    monitor = 'val_acc', save_best_only = True,
                                    save_weights_only=False,mode = 'auto',
                                    save_freq='epoch', verbose=1)
    csv_log = CSVLogger(csv_path, append=False)

    model.fit(training_data, validation_data= validate_data, epochs = 30,
               verbose = 1, callbacks=[csv_log,model_checkpoint_callback])



    params = {'X_train' : X_train,
              'Y_train' :Y_train,
              'Y_test' : Y_test,
              'X_test' : X_test,
              'batch_size' : batch_size}

    model.save('C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/models/{}.h5'.format(model_selection))

    
##    model = CONVOLUTION2D(no_classes = 2, sample_rate = 16000, sample_time = 1.0, **params)

        
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Data Training')
    parser.add_argument('--model_selection', type=str, default = 'Alexnet',
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
    
