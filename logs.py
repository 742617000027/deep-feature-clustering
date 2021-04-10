import sys

sys.path.append('..')

import pandas as pd
import matplotlib
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hparams import hparams as hp


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_image_data(data):
    fig, axs = plt.subplots(len(data), 1, figsize=(12, len(data) * 3), constrained_layout=True)

    for i, (key, value) in enumerate(data.items()):
        if value['type'] == 'spectrogram':
            im = axs[i].imshow(value['data'],
                               aspect='auto',
                               origin='lower',
                               interpolation='none')
            im.set_clim(hp.dsp.min_vol, hp.dsp.max_vol)
            fig.colorbar(im, ax=axs[i])
            axs[i].set_title(value['title'], loc='left')
            axs[i].set_xlabel('Frames')
            axs[i].set_ylabel('Bins')

        if value['type'] == 'mask':
            im = axs[i].imshow(value['data'],
                               aspect='auto',
                               origin='lower',
                               interpolation='none',
                               cmap='coolwarm')
            im.set_clim(-value['lim'], value['lim'])
            fig.colorbar(im, ax=axs[i])
            axs[i].set_title(f'{value["title"]} ({hp.training.mask_type})', loc='left')
            axs[i].set_xlabel("Frames")
            axs[i].set_ylabel("Bins")

        if value['type'] == 'wave':
            axs[i].plot(value['data'])
            axs[i].set_xlim([0, len(value['data'])])
            axs[i].set_title(value['title'], loc='left')
            axs[i].set_xlabel("Samples")
            axs[i].set_ylabel("Amplitude")

        if value['type'] == 'embedding':
            im = axs[i].imshow(value['data'],
                               aspect='auto',
                               origin='lower',
                               interpolation='none',
                               cmap='bone')
            fig.colorbar(im, ax=axs[i])
            axs[i].set_title(value['title'], loc='left')
            axs[i].axis('off')

    fig.canvas.draw()
    fig_data = save_figure_to_numpy(fig)
    plt.close()
    return fig_data


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)
        self.scalars = dict()

    def log_training(self, step, scalars):
        for plot, series in scalars.items():
            self.add_scalars(plot, series, step)

    def log_validation(self, model, step, scalars, image_data=None, embedding_data=None, datapoint_losses=None):
        for plot, series in scalars.items():
            self.add_scalars(plot, series, step)

        # Log and plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), global_step=step)

        # Log image data
        if image_data is not None:
            for i, elem in enumerate(image_data):
                self.add_image(f'{i}', plot_image_data(elem), global_step=step, dataformats='HWC')

        # Log embeddings
        if embedding_data is not None:
            lookup = {
                '01 Kicks': self.img([0.1, 0.1, 0.1]),
                '02 Snares': self.img([0.5, 0.5, 0.5]),
                '03 Claps': self.img([238 / 255, 136 / 255, 102 / 255]),
                '04 Hats': self.img([255 / 255, 255 / 255, 102 / 255]),
                '05 Rides': self.img([255 / 255, 204 / 255, 0 / 255]),
                '06 Perc': self.img([102 / 255, 102 / 255, 255 / 255]),
                '07 Crashes': self.img([255 / 255, 153 / 255, 0 / 255]),
                '08 Toms': self.img([51 / 255, 204 / 255, 51 / 255]),
                '11 Drumcomputer': self.img([0.8, 0.8, 0.8]),
            }
            embedding = embedding_data[0]
            metadata = embedding_data[1]
            label_img = []
            for file in metadata:
                for drum, color in lookup.items():
                    if drum in file:
                        label_img.append(color)
                        break
            label_img = torch.stack(label_img)
            self.add_embedding(embedding, metadata=metadata, label_img=label_img, global_step=step)

        # Log losses
        if datapoint_losses is not None:
            sorted_losses = dict()
            for key, value in sorted(datapoint_losses.items(), key=lambda item: item[1], reverse=True):
                sorted_losses[key] = value
            df = \
                pd.DataFrame([[k.replace('../data', ''), v]
                              for k, v in sorted_losses.items()][:hp.training.n_entries_in_valid_loss_report],
                             columns=['path', 'loss'])
            self.add_text('ordered losses', df.to_markdown(), global_step=step)

    def img(self, color_value, H=10, W=10):
        return torch.tensor(color_value).unsqueeze(1).unsqueeze(2).repeat(1, H, W)
