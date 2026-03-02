# %% Import libraries
from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msa.feature_extraction.features import STFT, stft_parallel

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

# %% STFT Parameters
sr = 200
STF = STFT(sr, 45001*2, None, "boxcar")
times, freqs, Zxx = stft_parallel(X, STF, )
Sxx = np.abs(Zxx)#**2
#Sxx = np.real(Zxx)**2 # OPIAHSAIOHDOAIBDIOPASDBASIPDBASPDBASIPDBASPDBNASIPDBASPDBNAS
Zxx.shape, times.shape, freqs.shape
# %%
import torch
X_torch = torch.from_numpy(X)
res = torch.stft(X_torch, 45001, 45001, 45001, return_complex=False)

# %% Visualization
from msa.visualization import plot
fig, ax = plt.subplots(2,1, sharex=True)
sample = 50
ax[0].plot(X[sample, :])
_, _ , mesh = plot.spectrogram(times[:]*sr, freqs[0:-1:2], aux[sample,:,:], logscale=True, ax = ax[1]);

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("bottom", size="15%", pad=0.4)
fig.colorbar(mesh, cax=cax, orientation='horizontal');

# %% Calculate the FFT
X_FFT = np.abs(np.fft.rfft(X, axis=1))
# X_FFT = np.abs(np.fft.fft(X, axis=1))

X_FFT.shape
# X_FFT

# %% RAM saving: Filter away higher frequencies
n_freqs = len(freqs)
Sxx = Sxx[:,:n_freqs//1, :]
freqs = freqs[:n_freqs//1]
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

