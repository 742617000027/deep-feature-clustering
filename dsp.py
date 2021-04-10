import librosa
import numpy as np
import pyACA
import scipy.signal
import soundfile as sf

from hparams import hparams as hp


def load_audio(file):
    info = sf.info(file)
    audio, _ = sf.read(file, always_2d=True)
    if audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=1)

    if info.samplerate > hp.dsp.sample_rate:
        audio = librosa.core.resample(audio, info.samplerate, hp.dsp.sample_rate)

    # Normalize audio and scale to max_vol defined in hparams
    audio = librosa.util.normalize(audio) * librosa.db_to_amplitude(hp.dsp.max_vol)
    return audio


def minmax_scale(x, series_min, series_max, new_min=0., new_max=1.):
    return (x - series_min) / (series_max - series_min) * (new_max - new_min) + new_min


def spectrograms(audio):
    S_stereo = []
    M_stereo = []

    window = np.hanning(hp.dsp.n_fft)

    for ch in range(2):
        S = librosa.core.stft(audio[:, ch],
                              n_fft=hp.dsp.n_fft,
                              hop_length=hp.dsp.hop_length,
                              win_length=None,
                              window=window,
                              center=True)
        S = 2 * np.abs(S) / np.sum(window)
        mel_basis = librosa.filters.mel(sr=hp.dsp.sample_rate,
                                        n_mels=hp.dsp.num_mels,
                                        n_fft=hp.dsp.n_fft,
                                        norm=None)
        enorm = 1. / (np.sum(mel_basis, axis=1, keepdims=True) + 1e-8)
        mel_basis *= enorm
        M = np.dot(mel_basis, S)

        S_stereo.append(S)
        M_stereo.append(M)

    S_stereo = np.stack(S_stereo, axis=2)
    M_stereo = np.stack(M_stereo, axis=2)
    return S_stereo, M_stereo


def amplitude_to_db(X):
    X = librosa.amplitude_to_db(X, ref=1.0, top_db=hp.dsp.max_vol - hp.dsp.min_vol)
    X += np.abs(np.max(X)) + hp.dsp.max_vol
    return X


def audio_features(S):
    features = []
    for ch in range(2):
        centroid = pyACA.FeatureSpectralCentroid(S[:, :, ch], hp.dsp.sample_rate)
        crest = pyACA.FeatureSpectralCrestFactor(S[:, :, ch], hp.dsp.sample_rate)
        decrease = pyACA.FeatureSpectralDecrease(S[:, :, ch], hp.dsp.sample_rate)
        flatness = pyACA.FeatureSpectralFlatness(S[:, :, ch], hp.dsp.sample_rate)
        flux = pyACA.FeatureSpectralFlux(S[:, :, ch], hp.dsp.sample_rate)
        kurtosis = pyACA.FeatureSpectralKurtosis(S[:, :, ch], hp.dsp.sample_rate)
        rolloff = pyACA.FeatureSpectralRolloff(S[:, :, ch], hp.dsp.sample_rate)
        skewness = pyACA.FeatureSpectralSkewness(S[:, :, ch], hp.dsp.sample_rate)
        slope = pyACA.FeatureSpectralSlope(S[:, :, ch], hp.dsp.sample_rate)
        tpr = pyACA.FeatureSpectralTonalPowerRatio(S[:, :, ch], hp.dsp.sample_rate)
        features.append(np.stack([centroid, crest, decrease, flatness, flux, kurtosis, rolloff, skewness, slope, tpr]))
    return np.stack(features, axis=2)


def spectral_flux(this_frame, prev_frame):
    return np.sqrt(np.sum((this_frame - prev_frame) ** 2)) / len(this_frame)


def spectral_rolloff(frame):
    threshold = 0.95 * np.sum(frame ** 2)
    E = 0.
    k = 0
    while E < threshold and k < len(frame):
        E = np.dot(frame[:k], frame[:k])
        k += 1
    return k / len(frame)


def spectral_centroid(frame):
    if np.sum(frame) == 0:
        return 0.5
    return np.dot(np.arange(len(frame)), frame) / ((len(frame) - 1) * (np.sum(frame)))


def spectral_spread(frame, centroid):
    norm = np.sum(frame)
    if norm == 0:
        norm = 1.
    return np.sqrt(np.dot((np.arange(len(frame)) - centroid) ** 2, frame) / norm)


def spectral_flatness(frame):
    gmean = np.exp(np.mean(np.log(1. + frame ** 2))) - 1.
    amean = np.mean(frame ** 2)
    return gmean / (amean + 1e-8)


def spectral_crest(frame):
    norm = np.sum(frame)
    if norm == 0:
        norm = 1.
    return np.max(frame) / norm


def spectral_kurtosis(frame, centroid, spread):
    f = np.arange(0, len(frame)) / (len(frame) - 1) * hp.dsp.sample_rate / 2
    centroid *= hp.dsp.sample_rate / 2
    norm = np.sum(frame)
    if norm == 0:
        norm = 1.
    if spread == 0:
        spread = 1.
    kurtosis = np.dot((f - centroid) ** 4, frame) / (spread ** 4 * norm * len(frame))
    return kurtosis - 3


def spectral_skewness(frame, centroid, spread):
    f = np.arange(0, len(frame)) / (len(frame) - 1) * hp.dsp.sample_rate / 2
    centroid *= hp.dsp.sample_rate / 2
    norm = np.sum(frame)
    if norm == 0:
        norm = 1.
    if spread == 0:
        spread = 1.
    skewness = np.dot((f - centroid) ** 3, frame) / (spread ** 3 * norm * len(frame))
    return skewness


def spectral_slope(frame):
    k = np.arange(len(frame))
    return np.dot(k - np.mean(k), frame - np.mean(frame)) / np.sum((k - np.mean(k)) ** 2)


def tonal_power_ratio(frame, Gt=5e-4):
    frame_sum = np.sum(frame ** 2)
    if frame_sum < Gt:
        return 0.
    peaks = scipy.signal.find_peaks(frame ** 2, height=Gt)
    if not peaks[0].size:
        return 0.
    return np.sum(frame[peaks[0]] ** 2) / frame_sum


def diff(x, dim, mode='diff'):
    assert dim in range(len(x.shape)), 'dim must be an integer between 0 and the number of dimensions of x.'
    assert mode in ['same', 'diff'], 'mode must be "same" or "diff".'
    out = np.swapaxes(x, 0, dim)  # x.transpose(0, dim)
    out = out[1:] - out[:-1]
    if mode == 'same':
        out = np.concatenate([x[0].unsqueeze(0), out], axis=0)
    return np.swapaxes(out, 0, dim)


def transient_spectrum(audio):
    pass
