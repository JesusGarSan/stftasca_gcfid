# %% Import libraries
from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msa.feature_extraction.features import STFT, stft_parallel
import librosa
from airPLS import airPLS

# % Read data
F_data = pd.read_csv("data/Y_data.csv", header=None)
F_data_matrix = F_data.to_numpy()

idx = np.where(np.sum(F_data_matrix, axis=1)==0)

df = pd.read_csv("data/X_data.csv", header=None)
X = df.to_numpy()
X = np.delete(X, idx, axis=1).T
N_signals, signal_samples = X.shape

# %% Initial visualization
plt.plot(X[:,:].T, alpha=0.25);
# plt.ylim(-200, 2000)

# %% Baseline reduction using airPLS
print("Removing baseline...")
for sample_id in range(X.shape[0]):
    correction = airPLS(X[sample_id])
    X[sample_id] -= correction

# %% Visualization after baseline reduction
plt.plot(X[:,:].T, alpha=0.25);
# plt.ylim(-200, 2000)

# %% SNV - Autoscale row-wise
for i in range(X.shape[0]):
    aux = X[i, :]
    mu = np.nanmean(aux)
    sigma = np.nanstd(aux)
    X[i,:] = (aux - mu) /sigma
# %% Visualization after SNV reduction
plt.plot(X[:,:].T, alpha=0.25);
plt.ylim(-0.5, 5)

# %% Librosa STFT
sr = 200
win_length = X.shape[1] // 1
hop_length = win_length
D = librosa.stft(X, n_fft=win_length, hop_length=hop_length, win_length=win_length,
                 window="blackman", center=False)
Sxx = np.abs(D)**2
S_db = librosa.power_to_db(Sxx)
frequencies = librosa.fft_frequencies(sr=sr, n_fft=win_length)
times = librosa.frames_to_time(np.arange(D.shape[-1]), 
                                     sr=sr, 
                                     hop_length=hop_length)
D.shape

# % Visualization
sample = 11
fig, ax = plt.subplots(2,1, sharex=True, figsize=(10, 6))
# ax[0].plot(np.linspace(0,S_db.shape[-1]), X[sample])
ax[0].plot(X[sample])
ax[0].set_xlim(0,X.shape[1])

sample_indices = np.arange(S_db[sample].shape[1]) * hop_length + win_length//2

mesh = librosa.display.specshow(S_db[sample], sr=sr, n_fft=win_length, hop_length=hop_length, win_length=win_length,
                               x_axis='frames', x_coords=sample_indices,
                               y_axis='hz', cmap= 'nipy_spectral', ax=ax[1])

ax[1].set_ylim(0,10)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("bottom", size="7%", pad=0.5)
fig.colorbar(mesh, cax=cax, orientation='horizontal');
         
plt.tight_layout()
plt.show()

# %% Calculate the FFT
X_FFT = np.abs(np.fft.rfft(X, axis=1))
# X_FFT = np.fft.fft(X, axis=1)

X_FFT.shape
# X_FFT

# %% RAM saving: Filter away higher frequencies
freqs_id = np.where(frequencies<=10)

X_FFT = X_FFT[:, :np.max(freqs_id)+1]
Sxx = Sxx[:,:np.max(freqs_id)+1, :]
freqs = frequencies[freqs_id]
n_freqs = len(freqs)
# %% Testing with log
# Sxx = np.log(Sxx+1)

# %% Preprocessing - Sample scaling
X_sample_scaled = Sxx.copy();
for i in range(X_sample_scaled.shape[0]):
    aux = X_sample_scaled[i, :, :]
    mu = np.nanmean(aux)
    sigma = np.nanstd(aux)
    X_sample_scaled[i,:,:] = (aux - mu)# /sigma

# %% Preprocessing - Block scaling (windows)
X_block_scaled = Sxx.copy();
for i in range(X_block_scaled.shape[2]):
    aux = X_block_scaled[:, :, i]
    mu = np.nanmean(aux)
    sigma = np.nanstd(aux)
    X_block_scaled[:,:,-1] = (aux - mu)# /sigma

# %% Preprocessing - Sample Scaling & Block scaling (windows)
X_norm = Sxx.copy();
for i in range(X_norm.shape[0]):
    aux = X_norm[i, :, :]
    mu = np.nanmean(aux)
    sigma = np.nanstd(aux)
    X_norm[i,:,:] = (aux - mu)# /sigma

for i in range(X_norm.shape[2]):
    aux = X_norm[:, :, i]
    mu = np.nanmean(aux)
    sigma = np.nanstd(aux)
    X_norm[:,:,-1] = (aux - mu)# /sigma

# %% Unfolding
n_samples, n_freqs, n_times = Sxx.shape

X_raw = Sxx.copy().transpose(0, 2, 1)
X_raw = X_raw.reshape(n_samples, n_freqs*n_times)
# %%

X_sample_scaled = X_sample_scaled.transpose(0, 2, 1)
X_sample_scaled = X_sample_scaled.reshape(n_samples, n_freqs*n_times)

X_block_scaled = X_block_scaled.transpose(0, 2, 1)
X_block_scaled = X_block_scaled.reshape(n_samples, n_freqs*n_times)

X_norm = X_norm.transpose(0, 2, 1)
X_norm = X_norm.reshape(n_samples, n_freqs*n_times)

# %% Read complementary data
X_peak = pd.read_csv("data/X_peak.csv", header=None)
F_peak = pd.read_csv("data/Y_peak.csv", header=None)
F_data = pd.read_csv("data/Y_data.csv", header=None)
F_data = np.delete(F_data.to_numpy(), idx, axis = 0)

# %% Save data
data_dict = {
    "X_raw": X_raw,
    "X_sample_scaled": X_sample_scaled,
    "X_block_scaled": X_block_scaled,
    "X_norm": X_norm,
    "times": times,
    "freqs": freqs,
    "X_fft": X_FFT,

    "X_peak": X_peak,
    "F_peak": F_peak,
    "F_data": F_data,
}
savemat("data.mat", data_dict)

