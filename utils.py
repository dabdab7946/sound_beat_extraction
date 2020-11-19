

import os
import numpy as np
import pickle
import gzip




# fft 분석
def fft(sig, HZ=44100, NFFT=100000, FFT_LEN=None):
    ham_window = np.hamming(NFFT) + 1e-100  # 1e-100: for recovery, devide by non-zero value
    freq = np.fft.fftfreq(NFFT, 1 / HZ)
    stft = np.fft.fft(sig * ham_window)
    # ---------------------------------------------------------
    if FFT_LEN is not None:
        freq = freq[: FFT_LEN]
        stft = stft[: FFT_LEN]
    return freq, abs(stft), stft        # abs(stft)는 amplitude세기 비교할 때 유용



#  data를 gzip 형식으로 압축하여 저장
def save_data(obj, filename):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file)



#  gzip 형식으로 압축된 파일을 로드
def load_data(filename):
    file = gzip.GzipFile(filename, 'rb')
    obj = pickle.load(file)
    return obj




