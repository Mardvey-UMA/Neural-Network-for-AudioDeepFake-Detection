import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import torch
import torch.nn as nn
import numpy as np
import time
import librosa
import librosa.display

print('-' * 50)


def make_tensor(absolute_path):
    path = absolute_path
    scale, sr = librosa.load(path)
    filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=128)
    mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    trch = torch.from_numpy(log_mel_spectrogram)
    if log_mel_spectrogram.shape != (10, 87):
        delta = 87 - log_mel_spectrogram.shape[1]
    trch = torch.nn.functional.pad(trch, (0, delta))
    np_arr = trch.cpu().detach().numpy()
    return np_arr


def create_train_data(save_path, summary):
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(path, topdown=True):
        for dirname in dirnames:
            for (direcpath, direcnames, files) in os.walk(path + '/' + dirname, topdown=True):
                for file in files:
                    actual_path = path + '/' + dirname + '/' + file
                    name = '/tensor' + str(i) + '_' + dirname + '.png'
                    if not os.path.isfile(save_path + '/' + dirname + name):
                        fig = plt.figure(figsize=(3.1189, 3.1189))
                        librosa.display.specshow(make_tensor(actual_path))
                        plt.subplots_adjust(0, 0, 1, 1)
                        plt.savefig(save_path + '/' + dirname + name)
                        print(save_path + '/' + dirname + name)
                        print(str(round((i / summary) * 100)) + '%')
                        print((round((i / summary) * 100)) * '[]')
                        #clear_output(wait=True)
                    i += 1


path = 'D:\\data\\testing'
create_train_data('D:\data\img_data\\test_img', 816)

path = 'D:\\data\\validation'
create_train_data('D:\\data\\img_data\\val_img', 2245)

path = 'D:\\data\\training'
create_train_data('D:\\data\\img_data\\train_img', 10206)
