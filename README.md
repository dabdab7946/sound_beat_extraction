# Audio Sound Beat추출 및 FFT분석 AI모델 개발 <br>
<br>

## 목차 <br>
 - 개요 <br>
 - 프로세스 <br>
 - 데이터 전처리 <br>
 - beat separation: librosa <br>
 - rule-based beat extraction <br>
 - AI beat extraction <br>
 - prerequisites <br>
 <br>
 
## 개요 <br>
Audio Sound에서 Beat Part만 따로 추출해서 그 위치(time)와 특성을 파악 <br> 
Rule-based 방식으로 Beat type을 구분하고 unit sound 저장 <br>
Unit sound를 학습하여 AI 방식으로 Beat type 구분 <br>
※ Beat type만 추출해서 활용할 수 있는 분야는 한정될 수 있지만, 유사한 방식으로 사운드 분석하여 다양한 분야에 활용 가능 <br>
<br>

## 프로세스 <br>
1. 데이터전처리 (mp3 -> wav) <br>
2. Beat Extraction(Rule-base) <br>
3. Beats 단위데이터 수집 <br>
4. Training (FFT 데이터) <br>
5. Beats Extraction by AI <br>
<br>

## Prerequisites<br>
 - librosa <br>
 - tensorflow1.15 <br>
 - pydub, soundfile <br>
 - scipy <br>
 <br>
 
 
