import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

import dsp
from hparams import hparams as hp


class ClusterDataset(IterableDataset):
    def __init__(self, files, step=0, running_mean=None, running_std=None):
        self.files = files
        self.step = step
        self.running_mean = running_mean
        self.running_std = running_std

    def __iter__(self):
        rng = np.random.default_rng()

        while True:
            idx = rng.integers(len(self.files))
            file = self.files.iloc[idx]['path']
            audio = dsp.load_audio(file)
            S, M = dsp.spectrograms(audio)
            inputs = self.make_inputs(M)
            features = dsp.audio_features(S)
            M = dsp.amplitude_to_db(np.mean(M, axis=-1))
            yield self.tensor(inputs), self.tensor(features), self.tensor(audio), self.tensor(M), file

    def make_inputs(self, M):
        channels = []
        for ch in range(2):
            M_scaled = dsp.amplitude_to_db(M[:, :, ch])
            M_scaled = dsp.minmax_scale(M_scaled, hp.dsp.min_vol, hp.dsp.max_vol)
            M_deltas = dsp.diff(np.pad(M_scaled, ((0, 0), (1, 0))), dim=1)
            M_deltadeltas = dsp.diff(np.pad(M_deltas, ((0, 0), (1, 0))), dim=1)
            channels.append(np.stack([M_scaled, M_deltas, M_deltadeltas]))
        return np.concatenate(channels, axis=0)

    def normalize(self, features, validation=False):
        if self.running_mean is None and self.running_std is None:
            self.step += 1
            self.running_mean = torch.mean(features, dim=(1, 2)).requires_grad(False)
            self.running_std = torch.zeros(10).requires_grad(False)
            return torch.cat([features[:, :, :, 0], features[:, :, :, 1]], dim=1)
        else:
            if not validation:
                self.step += 1
                local_mean = torch.mean(features, dim=(1, 2)).requires_grad(False)
                new_mean = self.running_mean + (local_mean - self.running_mean) / self.step
                self.running_std = self.running_std + (local_mean - self.running_mean) * (local_mean - new_mean)
                self.running_mean = new_mean
            return torch.cat(
                [(features[:, :, :, 0] - self.running_mean) / self.running_std,
                 (features[:, :, :, 1] - self.running_mean) / self.running_std],
                dim=1
            )

    def tensor(self, x):
        return torch.tensor(x.astype(np.float32))


def dir_walk(path, ext):
    file_list = [os.path.join(root, name)
                 for root, dirs, files in os.walk(path)
                 for name in sorted(files)
                 if name.endswith(ext)
                 and not name.startswith('.')]
    return file_list


def parse_data_structure(path):
    # Checks if supplied path points to dir or dataframe
    assert path != '', 'Please set your data paths in hparams.py.'
    if os.path.isdir(path):
        files = dir_walk(path, ext=('.wav', '.WAV'))
        df = pd.DataFrame(files, columns=['path'])
        return df
    elif path.endswith('.pkl'):
        df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame):
            raise IOError(f'"{path}" does not point to a valid pandas DataFrame.')
        if 'path' not in df.columns:
            raise Exception(f'DataFrame at "{path}" does not contain required column "path".')
        return df
    else:
        raise IOError(f'Argument  path  must point to directory or pickled DataFrame, but points to "{path}".')
