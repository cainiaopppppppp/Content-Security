import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载音频文件
audio_path = 'D:/Content_Secu/task2/xm2512.wav'
y, sr = librosa.load(audio_path)

# 计算声谱图
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

# 计算MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 计算频谱质心
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

# 计算频谱平坦度
spectral_flatness = librosa.feature.spectral_flatness(y=y)

# 计算过零率
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

# 计算声谱能量
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

# 计算声谱对比度
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

# 计算声谱滚降点
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

# 绘图展示特征
plt.figure(figsize=(12, 11))

# 绘制声谱图
plt.subplot(4, 2, 1)
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')

# 绘制波形
plt.subplot(4, 2, 2)
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')

# 绘制MFCCs
plt.subplot(4, 2, 3)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCCs')

# 绘制频谱质心
plt.subplot(4, 2, 4)
librosa.display.specshow(spectral_centroids, sr=sr, x_axis='time')
plt.colorbar()
plt.title('Spectral Centroids')

# 绘制频谱平坦度
plt.subplot(4, 2, 5)
librosa.display.specshow(spectral_flatness, x_axis='time')
plt.colorbar()
plt.title('Spectral Flatness')

# 绘制过零率
plt.subplot(4, 2, 6)
plt.plot(zero_crossing_rate[0])
plt.title('Zero Crossing Rate')

# 绘制声谱对比度
plt.subplot(4, 2, 7)
librosa.display.specshow(spectral_contrast, sr=sr, x_axis='time')
plt.colorbar()
plt.title('Spectral Contrast')

plt.tight_layout()
plt.savefig('1.png')
plt.show()
