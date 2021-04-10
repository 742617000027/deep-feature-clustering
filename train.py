import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoints import EarlyStopping, load_checkpoint
from data import ClusterDataset, parse_data_structure
from hparams import hparams as hp
from logs import Logger
from model import AutoEncoder


def inference(model, dataset):
    image_data = []

    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(hp.training.n_inference_examples), desc='Inference'):
            inputs, features, audio, M, file = next(iter(dataset))

            inputs = inputs.to(0, non_blocking=True).unsqueeze(0)
            features = dataset.normalize(features.unsqueeze(0)).to(0, non_blocking=True)
            M = M.to(0, non_blocking=True).unsqueeze(0)

            with autocast():
                y, encoding = model(inputs, features)

            image_data.append({
                'original_sample': {
                    'title': file.replace('../data', ''),
                    'type': 'spectrogram',
                    'data': M.detach().cpu().numpy().astype(np.float32).squeeze()
                },
                'reconstructed_sample': {
                    'title': 'Reconstructed Sample',
                    'type': 'spectrogram',
                    'data': y.detach().cpu().numpy().astype(np.float32).squeeze()
                },
                'audio': {
                    'title': 'Waveform',
                    'type': 'wave',
                    'data': audio.detach().cpu().numpy().astype(np.float32).squeeze()
                },
                'encoding': {
                    'title': 'Encoding',
                    'type': 'embedding',
                    'data': encoding.detach().cpu().numpy().astype(np.float32)
                }
            })

    return image_data


def validation(model, criterion, loader, dataset):
    valid_loss = []
    data = dict()

    model.eval()
    with torch.no_grad():
        with tqdm(desc='Valid', total=hp.training.n_validation_steps) as pbar:
            for inputs, features, _, M, file in loader:

                if pbar.n >= hp.training.n_validation_steps:
                    break

                inputs = inputs.to(0, non_blocking=True)
                features = dataset.normalize(features).to(0, non_blocking=True)
                M = M.to(0, non_blocking=True)

                with autocast():
                    y, encoding = model(inputs, features)
                    valid_loss.append(criterion(y, M).item())

                    data[file[0]] = {
                        'loss': valid_loss[-1],
                        'encoding': encoding.squeeze()
                    }

                pbar.set_postfix(loss=valid_loss[-1])
                pbar.update()

            valid_loss = np.mean(valid_loss)

            embedding = torch.stack([v['encoding'] for v in data.values()]).detach().cpu().numpy().astype(np.float32)
            return valid_loss, \
                   (embedding, list(data.keys())), \
                   {k: data[k]['loss'] for k in data}


def training(model, optimizer, scaler, criterion, loader, dataset, early_stopping, logger):
    global step

    with tqdm(desc='Train') as pbar:
        pbar.update(step)
        for inputs, features, _, M, _ in loader:

            inputs = inputs.to(0, non_blocking=True)
            features = dataset.normalize(features).to(0, non_blocking=True)
            M = M.to(0, non_blocking=True)

            model.train()
            with autocast():
                y, _ = model(inputs, features)
                loss = criterion(y, M) / hp.training.accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % hp.training.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()

            pbar.set_postfix(loss=loss.item(), patience=f'{early_stopping.counter}/{hp.training.patience}')
            pbar.update()
            step = pbar.n

            logger.log_training(pbar.n, {
                'training.loss': {'loss': loss.item()},
            })

            if pbar.n % hp.training.validation_every_n_steps == 0:
                valid_loss, embedding_data, datapoint_losses = validation(model, criterion, loader, dataset)
                image_data = inference(model, dataset)
                logger.log_validation(model,
                                      step=pbar.n,
                                      scalars={'validation.loss': {'loss': valid_loss}},
                                      image_data=image_data,
                                      embedding_data=embedding_data,
                                      datapoint_losses=datapoint_losses)
                early_stopping(valid_loss, model, optimizer, scaler, step)

                if early_stopping.early_stop:
                    break


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint file to continue training from.')
    args = parser.parse_args()

    # Initialize torch modules
    model = AutoEncoder()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.training.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.MSELoss()

    # Load parameters from checkpoint to resume training or start new training
    if args.checkpoint is not None:
        run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        model, optimizer, scaler, early_stopping_score, early_stopping_counter, running_mean, running_std, step, = \
            load_checkpoint(run_dir, model, optimizer, scaler, 'latest')
    else:
        now = time.strftime("%Y-%m-%d__%H_%M_%S", time.localtime())
        run_dir = os.path.join('..', 'runs', now)
        os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
        early_stopping_score = None
        early_stopping_counter = 0
        step = 0

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(step=step,
                                   run_dir=run_dir,
                                   best_score=early_stopping_score,
                                   counter=early_stopping_counter,
                                   verbose=False)

    # Initializing logger
    logger = Logger(os.path.join(run_dir, 'logs'))

    # Initialize data loaders
    dataset = ClusterDataset(files=parse_data_structure(hp.files),
                             feature_mean=np.load('mu.npy'),
                             feature_std=np.load('std.npy'))
    loader = DataLoader(dataset=dataset,
                        batch_size=hp.training.batch_size,
                        num_workers=hp.training.num_workers,
                        pin_memory=True)

    # Auto select best algorithm to maximize GPU utilization
    cudnn.benchmark = True

    # Execute train loop
    # try:
    training(model, optimizer, scaler, criterion, loader, dataset, early_stopping, logger)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    # save_checkpoint(run_dir, model, optimizer, scaler, early_stopping_score, early_stopping_counter, dataset, step, 'latest')
    # print('Saved checkpoint.')
