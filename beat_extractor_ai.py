

import os
import sys
import time
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
from pydub import AudioSegment


import tensorflow as tf



import models
import utils




# ====================================================================================
# Variables

AUDIO_PATH = 'dataset/'
MODEL_PATH = 'models/'


UNIT_TIME = 10      # 10초
THRESH = 0.1        #
hop_length = 512    # 전체 frame 수 (FFT에 맞춰야 하며 Default: 512)




# ====================================================================================
# Models
NUM_CLASSES = 2
C_LAYERS = 3
D_LAYERS = 2
KERNEL = 4
batch_size = 1000
Device = ''
SIG_SIZE = 2048
FFT_SIZE = 512


learning_rate = 1e-4


# Make Network & Train
PROJECT = 'beats'
g = tf.Graph()
with g.as_default():
    model = models.BeatClassifier(g, MODEL_PATH + '{}.ckpt'.format(PROJECT), Device,
            FFT_SIZE=FFT_SIZE, n_classes=NUM_CLASSES,
            C_LAYERS=C_LAYERS, D_LAYERS=D_LAYERS, KERNEL=KERNEL, learning_rate=learning_rate)






# ==================================================================================
# TEST

with open('./config.json', 'rb') as f:
    config = json.loads(f.read().decode())


audio_path = config.get('audio_path')
audio_file = config.get('audio_file')



# Load Music
middle = time.time()
sr = librosa.get_samplerate(audio_path + audio_file)
sound, sr = librosa.load(audio_path + audio_file, sr=sr)
duration = int(librosa.get_duration(sound, sr))


# Beat & Melody Separation
N_UNITS = int(duration / UNIT_TIME) + 1     # 단위시간으로 나눈 구간


print('>>> duration(sec): ', duration, ', sample_rate: ', sr, 'n_units: ', N_UNITS)




# 1 ~ END
beats_low_total = []            # 저음 Beat 수집
beats_high_total = []           # 고음 Beat 수집


for i in range(0, N_UNITS):
    unit_sound = sound[UNIT_TIME * i * sr: UNIT_TIME * (i + 1) * sr]
    UNIT_START_TIME = UNIT_TIME * i
    #
    D = librosa.stft(unit_sound)
    D_instruments, D_beats = librosa.decompose.hpss(D, margin=1)    # Instrument파트, Beat파트 구분
    unit_beats = librosa.istft(D_beats)                             # Beat파트 복원
    #
    onset_env = librosa.onset.onset_strength(unit_sound, sr=sr, aggregate=np.median)
    onset_env = librosa.util.normalize(onset_env)
    onset_env[onset_env < THRESH] = 0                               # 너무 작은 소리는 무시
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    #
    # ---------------------------------------------------------------------
    # BEAT Exctraction
    BEAT_START = 0
    BEAT_MAX = 0
    BEAT_MAX_VAL = 0
    LEN_UNIT_SOUND = len(onset_env)
    #
    for v in range(LEN_UNIT_SOUND):
        val = onset_env[v]
        unit_time = times[v]
        if val == 0 and BEAT_START == 0:
            continue
        # ----------------------------------------------------------------
        if val > 0 and BEAT_START == 0:
            BEAT_START = UNIT_START_TIME + unit_time
            BEAT_MAX = BEAT_START
            BEAT_MAX_VAL = val
        #
        elif val > 0 and BEAT_START > 0:
            if val > BEAT_MAX_VAL:
                BEAT_MAX_VAL = val
                BEAT_MAX = UNIT_START_TIME + unit_time
        #
        elif val == 0 and BEAT_START > 0:
            BEAT_END = UNIT_START_TIME + unit_time
            beat_type = 0
            # -------------------------------------------------------------------------------
            # FFT Analysis : too much time consumed!!!
            unit_beat_start = int(np.max([(BEAT_START - UNIT_START_TIME) - 0, 0]) * sr)
            unit_beat_end = int((BEAT_END - UNIT_START_TIME + 0) * sr)
            print('unit_beat:', np.round(BEAT_START, 3), '~', np.round(BEAT_END, 3))
            beat_sound = unit_beats[unit_beat_start: unit_beat_end]
            #
            # -------------------------------------------------------------------------------
            # Model 적용
            freq, fft_abs, fft_beat = utils.fft(beat_sound, len(beat_sound), len(beat_sound))
            fft_padding = np.zeros(FFT_SIZE)
            fft_padding[: len(fft_beat)] = fft_beat[: np.min([FFT_SIZE, len(fft_beat)])]
            # -------------------------------------------------------------------------------
            logit, pred = model.test(fft_padding.reshape(-1, FFT_SIZE, 1))
            print(logit, pred)
            if pred[0] == 0:
                beats_low_total += list(unit_sound)
            else:
                beats_high_total += list(unit_sound)
            # =============================================================================================
            BEAT_START = 0
            BEAT_MAX = 0
            BEAT_MAX_VAL = 0



# ---------------------------------------------------------------------------------------------
# Wav 만들기
save_beat_low_wav = audio_path + '{}_low_sounds.wav'.format(audio_file[: -4])
soundfile.write(save_beat_low_wav, np.array(beats_low_total), sr, format='WAV', endian='LITTLE',
                subtype='PCM_16')  # 깨지지 않음


save_beat_high_wav = audio_path + '{}_high_sounds.wav'.format(audio_file[: -4])
soundfile.write(save_beat_high_wav, np.array(beats_high_total), sr, format='WAV', endian='LITTLE',
                subtype='PCM_16')  # 깨지지 않음





