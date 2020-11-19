
######################## -*- coding: utf-8 -*-


import numpy as np
import os
import random
from scipy import signal    # resampling용


import librosa              # sound 처리용 package
import tensorflow as tf     # AI학습용 기본 package
import models               # Model Class 정의


import utils                # 유용한 utility 함수 정의



# ===============================================================================
# Environmental Variables
#
# [ Directory 구조 ]
# BeatExtractor/  (source codes here)
#   |----- dataset/
#           |----- train/
#   |----- models/



DATA_PATH = 'dataset/'
MODEL_PATH = 'models/'




FFT_SIZE = 512          # 학습에 반영할 FFT length (beat별로 길이가 다를 수 있는데 512Hz까지만 반영)
NUM_CLASSES = 2         # Heavy, High
C_LAYERS = 3            # Num of convolution layers
D_LAYERS = 2            # Num of dense layers (network after flattening)
KERNEL = 4
batch_size = 1000       # training batch size
Device = ''             # Multi-GPU 이용시 GPU No. 설정



# ===============================================================================
# 학습용 Dataset
class Dataset():
    def __init__(self, file_path, balance=True, balance_size=1000, augment=0):
        #self.sounds = []
        self.ffts = []
        self.labels = []
        heavy_files = os.listdir(file_path + 'bass/')
        high_files = os.listdir(file_path + 'high/')
        # light_files = os.listdir(file_path + 'beats_light/')
        # ----------------------------------------------------------------------------------
        # Balance: Training 및 중간 Validation용. Positive, Negative 숫자를 동일하게 밸런싱
        if balance:
            # Heavy
            for i in range(balance_size):
                index = random.randint(0, len(heavy_files) - 1)
                beat_file = file_path + '{}/{}'.format('bass/', heavy_files[index])
                if '.wav' not in beat_file:
                    continue
                sr = librosa.get_samplerate(beat_file)
                sound, sr = librosa.load(beat_file, sr=sr)
                duration = librosa.get_duration(sound, sr)
                # print(beat_file, sr, duration, len(sound))
                #
                if sr != 48000:
                    # print('resampling:', beat_file, sr)
                    sound = signal.resample(sound, int(len(sound) * 48000 / sr))
                #
                # FFT
                freq, fft_abs, fft_beat = utils.fft(sound, len(sound), len(sound))
                fft_padding = np.zeros(FFT_SIZE)
                fft_padding[: len(fft_beat)] = fft_beat[: np.min([FFT_SIZE, len(fft_beat)])]
                #
                # self.sounds.append(sound_padding.reshape(len(sound), 1))
                self.ffts.append(fft_padding.reshape(FFT_SIZE, 1))
                self.labels.append([1, 0])
            #
            # High
            for i in range(balance_size):
                index = random.randint(0, len(high_files) - 1)
                beat_file = file_path + '{}/{}'.format('high/', high_files[index])
                if '.wav' not in beat_file:
                    continue
                sr = librosa.get_samplerate(beat_file)
                sound, sr = librosa.load(beat_file, sr=sr)
                duration = librosa.get_duration(sound, sr)
                # print(beat_file, sr, duration, len(sound))
                #
                if sr != 48000:
                    # print('resampling:', beat_file, sr)
                    sound = signal.resample(sound, int(len(sound) * 48000 / sr))
                #
                # FFT
                freq, fft_abs, fft_beat = utils.fft(sound, len(sound), len(sound))
                fft_padding = np.zeros(FFT_SIZE)
                fft_padding[: len(fft_beat)] = fft_beat[: np.min([FFT_SIZE, len(fft_beat)])]
                #
                self.ffts.append(fft_padding.reshape(FFT_SIZE, 1))
                self.labels.append([0, 1])
        #
        # No-balance: 최종 Validation 또는 Test용. Dataset을 있는 그대로 적용
        else:
            # Heavy
            for index in range(0, len(heavy_files)):
                beat_file = file_path + '{}/{}'.format('bass/', heavy_files[index])
                if '.wav' not in beat_file:
                    continue
                sr = librosa.get_samplerate(beat_file)
                sound, sr = librosa.load(beat_file, sr=sr)
                duration = librosa.get_duration(sound, sr)
                # print(beat_file, sr, duration, len(sound))
                #
                if sr != 48000:
                    # print('resampling:', beat_file, sr)
                    sound = signal.resample(sound, int(len(sound) * 48000 / sr))
                #
                # FFT
                freq, fft_abs, fft_beat = utils.fft(sound, len(sound), len(sound))
                fft_padding = np.zeros(FFT_SIZE)
                fft_padding[: len(fft_beat)] = fft_beat[: np.min([FFT_SIZE, len(fft_beat)])]
                #
                self.ffts.append(fft_padding.reshape(FFT_SIZE, 1))
                self.labels.append([1, 0])
            #
            # High
            for index in range(0, len(high_files)):
                beat_file = file_path + '{}/{}'.format('high/', high_files[index])
                if '.wav' not in beat_file:
                    continue
                sr = librosa.get_samplerate(beat_file)
                sound, sr = librosa.load(beat_file, sr=sr)
                duration = librosa.get_duration(sound, sr)
                # print(beat_file, sr, duration, len(sound))
                #
                if sr != 48000:
                    # print('resampling:', beat_file, sr)
                    sound = signal.resample(sound, int(len(sound) * 48000 / sr))
                #
                # FFT
                freq, fft_abs, fft_beat = utils.fft(sound, len(sound), len(sound))
                fft_padding = np.zeros(FFT_SIZE)
                fft_padding[: len(fft_beat)] = fft_beat[: np.min([FFT_SIZE, len(fft_beat)])]
                #
                self.ffts.append(fft_padding.reshape(FFT_SIZE, 1))
                self.labels.append([0, 1])
        #
        print('data len:', np.sum(self.labels, axis=0))
        # -------------------------------------------------------------------------------------
        self.ffts = np.array(self.ffts)
        self.labels = np.array(self.labels)




# =========================================================================================
# Load Models
learning_rate = 1e-4            # learning_rate 는 학습정도에 따라 조정 필요
PROJECT = 'beats_classifier'
g = tf.Graph()
with g.as_default():
    model = models.BeatClassifier(g, MODEL_PATH + '{}.ckpt'.format(PROJECT), Device,
            FFT_SIZE=FFT_SIZE, n_classes=NUM_CLASSES,
            C_LAYERS=C_LAYERS, D_LAYERS=D_LAYERS, KERNEL=KERNEL, learning_rate=learning_rate)




# =========================================================================================
def train_model(model, acc, dataset, batch_size=100):
    best_acc = acc
    print('' * 25)
    #
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            IS_TRAIN = True
        else:
            IS_TRAIN = False
        data = dataset.get(phase)
        if data is None:
            continue
        #
        # Training / Evaluation
        cost_sum = 0
        accuracy_sum = 0
        total_cnt = 0
        BATCH_CNT = int(len(data.ffts) / batch_size)
        for b in range(BATCH_CNT):
            batch_X = data.ffts[batch_size * b: batch_size * (b + 1)]
            batch_Y = data.labels[batch_size * b: batch_size * (b + 1)]
            #
            c, l, p, a = model.train(batch_X, batch_Y, IS_TRAIN)
            cost_sum += c
            accuracy_sum += a
            total_cnt += 1
        #
        # Statistics
        if total_cnt == 0:
            continue
        epoch_cost = cost_sum / total_cnt
        epoch_accuracy = accuracy_sum / total_cnt
        print(phase, ', cost={0:0.4f}'.format(epoch_cost), ', accuracy={0:0.4f}'.format(epoch_accuracy))
        #
        # Save Model
        if phase == 'val' and epoch_accuracy >= best_acc:
            print(model.ModelName)
            print('change best_model: {0:0.4f}'.format(epoch_accuracy))
            best_acc = epoch_accuracy
            model.save(model.ModelName)
            print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc




# ===============================================================================================================
# Training
best_acc = 0.3
LOAD_SIZE = 1000
num_epochs = 100000

for e in range(num_epochs):
    # Train data
    train_data = Dataset(DATA_PATH + 'train/', balance=True, balance_size=LOAD_SIZE * 2, augment=0)
    #
    # Valid data
    valid_data = Dataset(DATA_PATH + 'valid/', balance=True, balance_size=LOAD_SIZE, augment=0)
    #
    my_dataset = {'train': train_data, 'val': valid_data}
    # -------------------------------------------------------------------
    model, best_acc = train_model(model, best_acc, my_dataset, batch_size)
    # -------------------------------------------------------------------
    if e % 100 == 0:
        print('\nepoch:', e)
        print('best_acc:', best_acc)




