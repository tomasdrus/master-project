import librosa
import numpy as np
import matplotlib.pyplot as plt


#

path = 'VCTK-Corpus/wav48/p225/p225_001_mic1.flac'
path = 'VCTK-Corpus/wav48/p225/p225_003_mic1.flac'
y, sr = librosa.load(path, sr=48000, mono=True)
duration_orig = librosa.get_duration(y=y, sr=sr)

fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set(title='Envelope view, mono')
ax[0].label_outer()

y, _ = librosa.effects.trim(y, top_db=30, ref=np.max)
duration_trim = librosa.get_duration(y=y, sr=sr)
print(duration_orig, duration_trim)


librosa.display.waveshow(y, sr=sr, ax=ax[1])
ax[1].set(title='Envelope view, mono')
ax[1].label_outer()

""" y_harm, y_perc = librosa.effects.hpss(y)
librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax[1], label='Harmonic')
librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[1], label='Percussive')
ax[1].set(title='Multiple waveforms')
ax[1].legend() """
plt.show()

