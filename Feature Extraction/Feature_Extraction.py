import numpy as np
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os, os.path
from scipy.io.wavfile import read as wavread
import math


def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2 ** (nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return samplerate, audio


# A EXTRACTION
def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])

    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t


def fourier(x):
    # Get Symmetric fft
    w = signal.windows.hann(np.size(x))
    windowed = x * w
    w1 = int((x.size + 1) // 2)
    w2 = int(x.size / 2)
    fftans = np.zeros(x.size)

    # Centre to make even function
    fftans[0:w1] = windowed[w2:]
    fftans[w2:] = windowed[0:w1]
    X = fft(fftans)
    magX = abs(X[0:int(x.size // 2 + 1)])
    return magX


def extract_spectral_centroid(xb, fs):
    magX = np.zeros((xb.shape[0], int(xb.shape[1]/2 + 1)))
    centroids = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        magX[block] = fourier(xb[block])
        N = magX[block].size
        den = np.sum(magX[block])
        if den == 0:
            den = 1
        centroid = 0
        for n in range(N):
            num = magX[block][n] * n
            centroid += num / den
        centroid = (centroid / (N-1)) * fs/2
        centroids[block] = centroid
    return centroids


def extract_rms(xb):
    rms = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        rms[block] = np.sqrt(np.mean(np.square(xb[block])))
        threshold = 1e-5  # truncated at -100dB
        if rms[block] < threshold:
            rms[block] = threshold
        rms[block] = 20 * np.log10(rms[block])
    return rms


def extract_zerocrossingrate(xb):
    zcr = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        zcr[block] = 0.5 * np.mean(np.abs(np.diff(np.sign(xb[block]))))
    return zcr


def extract_spectral_crest(xb):
    magX = np.zeros((xb.shape[0], int(xb.shape[1]/2 + 1)))
    spc = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        magX[block] = fourier(xb[block])
        summa = np.sum(magX[block], axis=0)
        if not summa:
            summa = 1
        spc[block] = np.max(magX[block]) / summa
    return spc


def extract_spectral_flux(xb):
    magX = np.zeros((xb.shape[0], int(xb.shape[1]/2 + 1)))
    specflux = np.zeros((xb.shape[0]))
    magX[0] = fourier(xb[0])
    for block in np.arange(1, xb.shape[0]):
        magX[block] = fourier(xb[block])
        den = magX[block].shape[0]
        specflux[block] = np.sqrt(np.sum(np.square(magX[block] - magX[block-1])))/den
    return specflux


def extract_features(x, blockSize, hopSize, fs):
    """
    # Wrapper function which returns 5xNum of Blocks feature Matrix
    """
    # block audio
    xb, timeInSecs = block_audio(x, blockSize, hopSize, fs)
    numBlocks = math.ceil(x.size / hopSize)
    all_features = np.zeros((5, numBlocks))

    # extract all spectral features
    all_features[0] = extract_spectral_centroid(xb, fs)
    all_features[1] = extract_rms(xb)
    all_features[2] = extract_zerocrossingrate(xb)
    all_features[3] = extract_spectral_crest(xb)
    all_features[4] = extract_spectral_flux(xb)
    return all_features


def aggregate_feature_per_file(features):
    """
    Aggregates features by Mean and Standard deviation to return a 10X1 aggregate feature matrix
    """

    L_mean = [np.mean(features[i]) for i in range(features.shape[0])]
    L_std = [np.std(features[i]) for i in range(features.shape[0])]
    L_mean = np.asarray(L_mean)
    L_std = np.asarray(L_std)
    aggregated_feature_matrix = np.concatenate((L_mean, L_std))
    return aggregated_feature_matrix


def get_feature_data(path, blockSize, hopSize):
    """
    Loops over all the files contained within a folder
    call extract_feature() and aggregate_feature_per_file()
    to return a 10XN matrix which contains the aggregated features of N Audio files
    """
    num_of_files = len([name for name in os.listdir(path)])
    aggregated_feature_matrix = np.zeros((10, num_of_files))
    i = 0
    for file in os.listdir(path):
        if file.endswith('.wav'):
            file_name = os.path.join(path,file)
            fs, x = ToolReadAudio(file_name)
            file_features = extract_features(x, blockSize, hopSize, fs)
            aggregated_features = aggregate_feature_per_file(file_features)  # 10x1
            aggregated_feature_matrix[:, i] = aggregated_features
            i += 1

    return aggregated_feature_matrix


# B NORMALIZATION
def normalize_zscore(featureData):
    """
    zscore normalization scheme to the input feature matrix
    """
    normalizedFeatureData = np.empty_like(featureData)
    for i in range(featureData.shape[0]):
        normalizedFeatureData[i] = (featureData[i] - np.mean(featureData[i])) / np.std(featureData[i])
    return normalizedFeatureData


# C Visualization
def visualize_features(path_to_musicspeech):
    """
    Two seperate feature matrices for the files in each music and speech folders
    1. SC mean, SCR mean
    2. SF mean ZCR mean
    3. RMS mean RMS std
    4. ZCR std SCR std
    5. SC std SF std
    """
    blockSize = 1024
    hopSize = 256
    for folder in os.listdir(path_to_musicspeech):
        if folder.endswith('speech_wav'):
            folder_path = os.path.join(path_to_musicspeech, folder)
            speech_features = get_feature_data(os.path.join(folder_path), blockSize, hopSize)
        elif folder.endswith('music_wav'):
            folder_path = os.path.join(path_to_musicspeech, folder)
            music_features = get_feature_data(os.path.join(folder_path), blockSize, hopSize)
    all_features = np.hstack((speech_features, music_features))
    normalizedFeatureData = normalize_zscore(all_features)
    normalizedSpeechFeatures = normalizedFeatureData[:, :speech_features.shape[1]]
    normalizedMusicFeatures = normalizedFeatureData[:, music_features.shape[1]:]

    # Visualize features:
    plt.figure(figsize=(15, 15))
    plt.suptitle("Feature Visualization",fontsize=24)

    # SC Mean vs SCR Mean
    ax1 = plt.subplot(321)
    ax1.scatter(normalizedSpeechFeatures[0, :], normalizedSpeechFeatures[3, :], c='b', label='Speech', s=10)
    ax1.scatter(normalizedMusicFeatures[0, :], normalizedMusicFeatures[3, :], c='r', label='Music', s=10)
    ax1.title.set_text("SC Mean vs SCR Mean")
    ax1.set_xlabel("SC mean")
    ax1.set_ylabel("SCR mean")
    plt.legend()

    # SF mean ZCR mean
    ax2 = plt.subplot(322)
    ax2.scatter(normalizedSpeechFeatures[4, :], normalizedSpeechFeatures[2, :], c='b', label='Speech', s=10)
    ax2.scatter(normalizedMusicFeatures[4, :], normalizedMusicFeatures[2, :], c='r', label='Music', s=10)
    ax2.title.set_text("SF mean vs ZCR mean")
    ax2.set_xlabel("SF mean")
    ax2.set_ylabel("ZCR mean")
    plt.legend()

    # RMS mean RMS std
    ax3 = plt.subplot(323)
    ax3.scatter(normalizedSpeechFeatures[1, :], normalizedSpeechFeatures[6, :], c='b', label='Speech', s=10)
    ax3.scatter(normalizedMusicFeatures[1, :], normalizedMusicFeatures[6, :], c='r', label='Music', s=10)
    ax3.title.set_text("RMS mean vs RMS std")
    ax3.set_xlabel("RMS mean")
    ax3.set_ylabel("RMS std")
    plt.legend()

    # ZCR std SCR std
    ax4 = plt.subplot(324)
    ax4.scatter(normalizedSpeechFeatures[7, :], normalizedSpeechFeatures[8, :], c='b', label='Speech', s=10)
    ax4.scatter(normalizedMusicFeatures[7, :], normalizedMusicFeatures[8, :], c='r', label='Music', s=10)
    ax4.title.set_text("ZCR std vs SCR std")
    ax4.set_xlabel("ZCR std")
    ax4.set_ylabel("SCR std")
    plt.legend()

    # SC std SF std
    ax5 = plt.subplot(325)
    ax5.scatter(normalizedSpeechFeatures[5, :], normalizedSpeechFeatures[9, :], c='b', label='Speech', s=10)
    ax5.scatter(normalizedMusicFeatures[5, :], normalizedMusicFeatures[9, :], c='r', label='Music', s=10)
    ax5.title.set_text("SC std vs SF std")
    ax5.set_xlabel("SC std")
    ax5.set_ylabel("SF std")
    plt.tight_layout()
    plt.legend()
    plt.savefig('Plot.png')

    return 0