
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import json


import librosa
import soundfile



import utils



# =======================================================================================
# 파일 및 기타정보 설정

AUDIO_PATH = 'dataset/samples/'
audio_file = 'edm_02.wav'
SAVE_PATH = AUDIO_PATH + 'beats_{}/'.format(audio_file.split('.')[0])
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


UNIT_TIME = 10   #
THRESH = 0.15
hop_length = 512  # 전체 frame 수 (FFT에 맞춰야 하며 Default: 512)
MAX_HZ = 500





# =======================================================================================
# Unit Time Sound 분석 functiona
def find_beats(unit_sound, sr=44100, margin=1, THRESH=0.02, UNIT_START_TIME=0,
               audiopath='', audiofile='', show_chart=False, savefile=''):
    print('\tfind_beats:', audiopath, audiofile, UNIT_START_TIME, show_chart)
    unit_beat_list = []
    if len(unit_sound) == 0:
        return unit_beat_list
    # ---------------------------------------------------------------------
    D = librosa.stft(unit_sound)
    D_instruments, D_beats = librosa.decompose.hpss(D, margin=margin)       # instrument, beat 파트 분리 (voice는 나눠져서 들어감)
    unit_beats = librosa.istft(D_beats)                                     # beat파트만 복원
    #
    onset_env = librosa.onset.onset_strength(unit_sound, sr=sr, aggregate=np.median)
    onset_env = librosa.util.normalize(onset_env)
    onset_env[onset_env < THRESH] = 0                                       # 작은소리는 무시
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)     # beat 위치 찾기
    #
    # ---------------------------------------------------------------------
    # BEAT Exctraction
    BEAT_START = 0
    BEAT_MAX = 0
    BEAT_MAX_VAL = 0
    LEN_UNIT_SOUND = len(onset_env)
    #
    # unit_beat_heavy = []
    # unit_beat_high = []
    for v in range(LEN_UNIT_SOUND):
        val = onset_env[v]
        unit_time = times[v]
        if val == 0 and BEAT_START == 0:
            continue
        # -----------------------------------------------------------
        if val > 0 and BEAT_START == 0:
            BEAT_START = UNIT_START_TIME + unit_time
            BEAT_MAX = BEAT_START
            BEAT_MAX_VAL = val
        elif val > 0 and BEAT_START > 0:
            if val > BEAT_MAX_VAL:
                BEAT_MAX_VAL = val
                BEAT_MAX = UNIT_START_TIME + unit_time
        elif val == 0 and BEAT_START > 0:
            BEAT_END = UNIT_START_TIME + unit_time
            beat_type = 'bass'   # 0: low(bass)비트, 1: high비트
            # -------------------------------------------------------------------------------
            # FFT Analysis
            unit_beat_start = int(np.max([(BEAT_START - UNIT_START_TIME) - 0, 0]) * sr)
            unit_beat_end = int((BEAT_END - UNIT_START_TIME + 0) * sr)
            beat_sound = unit_beats[unit_beat_start: unit_beat_end]
            #
            freq, fft_abs, fft_beat = utils.fft(beat_sound, len(beat_sound), len(beat_sound))
            if np.max(fft_abs[150: 400]) < 2:           # fft 절대값 크기가 2보다 작은 경우 low
                # unit_beat_heavy += list(beat_sound)
                beat_type = 'bass'
            else:
                # unit_beat_high += list(beat_sound)
                beat_type = 'high'
            # -------------------------------------------------------------------------------
            new_beat = {
                'start': float(BEAT_START), 'end': float(BEAT_END),       # start-max-end 파형으로 beat속성 파악
                'max': float(BEAT_MAX), 'max_amplitude': float(BEAT_MAX_VAL),
                'beat_type': beat_type
            }
            unit_beat_list.append(new_beat)
            #
            # -------------------------------------------------------------------------------
            # Save Beat sound (origin)
            unit_beat_path = SAVE_PATH + 'unit_beats_{}/'.format(beat_type)
            if not os.path.exists(unit_beat_path):
                os.makedirs(unit_beat_path)
            #
            save_beat_wav = unit_beat_path + '{}_{}.wav'.format(audiofile.split('.')[0], '%.3f' % BEAT_START)
            soundfile.write(save_beat_wav, np.array(beat_sound), sr, format='WAV', endian='LITTLE', subtype='PCM_16')
            # -------------------------------------------------------------------------------
            # 새로운 beat찾기 위해 정보 초기화
            BEAT_START = 0
            BEAT_MAX = 0
            BEAT_MAX_VAL = 0
    #
    # ---------------------------------------------------------------------------------------------
    # Show Chart
    if show_chart == True:
        fig = plt.figure(figsize=(20, 15))
        temp = plt.subplot(3, 1, 1)
        unit_times = []
        for s in range(len(unit_sound)):
            unit_times.append(s / sr)
        temp = plt.plot(unit_times, unit_sound)
        temp = plt.xticks([])
        #
        temp = plt.subplot(3, 1, 2)
        unit_beat_indices = []
        for beat_info in unit_beat_list:
            max = beat_info.get('max') % UNIT_TIME
            unit_beat_indices.append(max)
        unit_beats = np.zeros(len(unit_sound))
        beat_temp = librosa.istft(D_beats)
        unit_beats[: len(beat_temp)] = beat_temp
        temp = plt.plot(unit_times, unit_beats)
        temp = plt.vlines(unit_beat_indices, -1, 1, color='r', alpha=0.9, linestyle='--', label='Beats')
        temp = plt.xticks([])
        #
        temp = plt.subplot(3, 1, 3)
        temp = plt.plot(times, onset_env)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        temp = plt.vlines(times[onset_frames], 0, 1, color='r', alpha=0.9, linestyle='--', label='Beats')
        #
        plt.savefig(savefile.format('%04d' % UNIT_START_TIME))
        plt.close()
    #
    return unit_beat_list




# ===========================================================================================
# Load Music
start = time.time()
sr = librosa.get_samplerate(AUDIO_PATH + audio_file)
sound, sr = librosa.load(AUDIO_PATH + audio_file, sr=sr)
duration = int(librosa.get_duration(sound, sr))


# Beat & Melody Separation
N_UNITS = int(duration / UNIT_TIME) + 1     # 단위시간으로 나눈 구간
print('>>> duration(sec): ', duration, ', sample_rate: ', sr, 'n_units: ', N_UNITS)



# ===========================================================================================
# Beat & Melody Separation

beat_info_list = []             # beat 분석정보 저장 => csv
for i in range(0, N_UNITS):     # 첫번째 UnitTime(10초) ~ END
    unit_sound = sound[UNIT_TIME * i * sr: UNIT_TIME * (i + 1) * sr]
    UNIT_START_TIME = UNIT_TIME * i
    #
    unit_beat_list = find_beats(
        unit_sound, sr, margin=1, THRESH=THRESH, UNIT_START_TIME=UNIT_START_TIME,
        show_chart=True, savefile=SAVE_PATH + 'unit_sound_{}.png', audiopath=AUDIO_PATH, audiofile=audio_file)
    #
    beat_info_list += unit_beat_list



# ===========================================================================================
# beat정보 csv 저장
csv_file = 'beats_{}.csv'.format(audio_file.split('.')[0])
csv_writer = open(SAVE_PATH + csv_file, 'w', encoding='utf-8')
csv_writer.write('{},{}'.format(audio_file, THRESH))
csv_writer.write('\ntime,amplitude,type')
for beat in beat_info_list:
    csv_writer.write('\n{},{},{}'.format(beat.get('max'), beat.get('max_amplitude'), beat.get('beat_type')))


csv_writer.flush()
csv_writer.close()


