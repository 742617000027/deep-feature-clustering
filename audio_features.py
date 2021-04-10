from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import dsp
from data import dir_walk
from hparams import hparams as hp


def process(file):
    audio = dsp.load_audio(file)
    S, _ = dsp.spectrograms(audio)
    features = dsp.audio_features(S)
    return np.mean(features, axis=(1, 2))


if __name__ == '__main__':

    files = dir_walk(hp.files, ('.wav', '.WAV'))
    with Pool(8) as pool:
        vals = []
        for val in tqdm(pool.imap_unordered(process, files), total=len(files)):
            vals.append(val)

    mat = np.stack(vals, axis=1)
    mu = np.mean(mat, axis=1)
    std = np.std(mat, axis=1)
    np.save('mu.npy', mu)
    np.save('std.npy', std)
