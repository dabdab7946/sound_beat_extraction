
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
import time






# =======================================================================================
# MP3 => WAV 변환
AUDIO_PATH = 'dataset/'

for dirname, subdirs, files in os.walk(AUDIO_PATH):
    for audio_file in files:
        if audio_file.split('.')[-1].lower() in ['mp3']:
            start = time.time()
            #
            sound = AudioSegment.from_mp3(dirname + '/' + audio_file)
            audio_file = '{}.wav'.format(audio_file.split('.')[0])
            sound.export(dirname + '/' + audio_file, format='wav')
            print(audio_file, '소요시간:', time.time() - start)



