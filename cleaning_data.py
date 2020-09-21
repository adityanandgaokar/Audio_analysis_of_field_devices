import argparse
import os
import glob
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
import librosa
from librosa.core import resample,to_mono
import pandas as pd
import matplotlib.pyplot as plt
def checking_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def downsample(path, down_sample):
    sample_rate, wave = wavfile.read(path)
    wave = wave.astype(np.float32, order = 'F')

##    wave, sample_rate = librosa.load(path, sr = args.down_sample, mono=True)
    
    try:
        tmp = wave.shape[1]
        wave = to_mono(wave.T)
    except:
        pass
    wave = resample(wave, sample_rate, down_sample)
    wave = wave.astype(np.int16)

    return  wave, down_sample

def envelope(ver, sample_rate, threshold):
    mask = []
    ver = pd.Series(ver).apply(np.abs)
    ver_mean = ver.rolling(window = int(sample_rate/20), min_periods = 1,
                               center= True).max()
    for base in ver_mean:
        if base < threshold:
            mask.append(False)
        else:
            mask.append(True)
    return mask, ver_mean

def saving_sample(sample, sample_rate, end_dir, file_name, no):
    file_name = file_name.split('.wav')[0]
    target_path = os.path.join(end_dir.split('.')[0], file_name +'_{}.wav'.format(str(no)))

    if os.path.exists(target_path):
        return
    wavfile.write(target_path, sample_rate, sample)
    

def breaking_files(args):
    source_fls = args.source_fls
    des_fls = args.des_fls
    wave_files = glob.glob('{}/**'.format(source_fls), recursive= True)
    #print(wave_files)
    directories = os.listdir(source_fls)
    #print(directories)
    checking_dir(des_fls)
    classes = os.listdir(source_fls)
    #print(classes)
    for cls in classes:
        end_dir = os.path.join(des_fls, cls)
        checking_dir(end_dir)
        #print(end_dir)
        source_direc = os.path.join(source_fls, cls)
        #print(source_direc)
        for file_name in tqdm(os.listdir(source_direc)):
            source_file_name = os.path.join(source_direc, file_name)
            #print(source_file_name)
            sample_rate, wave = downsample(source_file_name, args.down_sample)
            mask, ver_mean = envelope(wave, sample_rate, threshold=args.threshold)
            wave = wave[mask]
            S = args.sample_time
            sub_sample =  int(S*sample_rate)

            ## stepping and saving every sample if sample is too small then discarding

            if wave.shape[0] > sub_sample:
                cutting = wave.shape[0] % sub_sample
                for count, j in enumerate(np.arange(0, wave.shape[0]- cutting, sub_sample)):
                    start = int(j)
                    stop = int(j + sub_sample)
                    sample = wave[start:stop]
                    saving_sample(sample, sample_rate, end_dir, file_name, count)

                    
            ## sample which is less than delta_sample pad with zeros

            else:
                sample = np.zeros(shape = (sub_sample), dtype = np.int16)
                sample[:wave.shape[0]] = wave
                saving_sample(sample, sample_rate, end_dir, file_name, 0)

                

def checking_threshold(args):
    source_fls = args.source_fls
    wave_paths = glob.glob('{}/**'.format(source_fls), recursive=True)
    
    wav_file = [x for x in wave_paths if args.file_name in x]

    if len(wav_file) == 1:
        sample_rate, wave = downsample(wav_file[0], args.down_sample)
        mask, envel = envelope(wave, sample_rate, threshold=args.threshold)
        plt.style.use('ggplot')
        plt.title('Signal Envelope, Threshold = {}'. format(str(args.threshold)))
        #plt.plot(wave[np.logical_not(mask)], color='r', label='remove')
        plt.plot(wave[mask], color='c', label= 'keep')
        plt.plot(envel, color='m', label= 'envelope')
        plt.grid(False)
        plt.legend(loc='best')
        plt.show()
    else:
        print('audio file {} is not in substring'. format(args.file_name))
        return
    
    
if __name__ == '__main__':

    infer = argparse.ArgumentParser(description='clean audio files')

    infer.add_argument('--source_fls', type=str, default = 'C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/wave_files/trail',
                           help = 'audio directory files of total duration')
    infer.add_argument('--des_fls', type=str, default = 'C:/Users/i00504285/Desktop/Aditya/Sound_Analysis/CSF48/wave_files/acc_audio/',
                           help = 'audio directory separated in order to delta time(clean files)')
    infer.add_argument('--sample_time', '-S', type= float, default= '31.0',
                           help = 'sample the audio for perticular second')
    infer.add_argument('--down_sample', type=int, default=16000,
                           help = 'down sampling audio')
    infer.add_argument('--file_name', type= str, default= '7ac9dad1-bb96-49bb-9c84-a30a430b2a31.wav',
                           help = 'file to plot over time to check magnitude')
    infer.add_argument('--threshold', type=int, default='100',
                           help = 'threshold for amplitude')

    args, _ = infer.parse_known_args()

    #breaking_files(args)
    checking_threshold(args)
