import os

import torch

from hparams import hparams as hp


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, step, run_dir, best_score=None, counter=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.step = step
        self.run_dir = run_dir
        self.patience = hp.training.patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, scaler, dataset, step):

        self.step = step
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            save_checkpoint(self.run_dir, model, optimizer, scaler,
                            self.best_score, self.counter, dataset, step, 'best')
            save_checkpoint(self.run_dir, model, optimizer, scaler,
                            self.best_score, self.counter, dataset, step, 'latest')
        elif score < self.best_score:
            self.counter += 1
            save_checkpoint(self.run_dir, model, optimizer, scaler,
                            self.best_score, self.counter, dataset, step, 'latest')
            if self.verbose:
                print(f'SynthStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_checkpoint(self.run_dir, model, optimizer, scaler,
                            self.best_score, self.counter, dataset, step, 'best')
            save_checkpoint(self.run_dir, model, optimizer, scaler,
                            self.best_score, self.counter, dataset, step, 'latest')
            self.counter = 0


def save_checkpoint(run_dir, model, optimizer, scaler, early_stopping_score, early_stopping_counter,
                    dataset, step, best_or_latest, save_running=False):
    assert best_or_latest in ['best', 'latest'], 'best_or_latest must be "best" or "latest".'
    pt = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'early_stopping_score': early_stopping_score,
        'early_stopping_counter': early_stopping_counter,
        'running_mean': dataset.running_mean,
        'running_std': dataset.running_std,
        'step': step,
    }
    torch.save(pt, os.path.join(run_dir, f'checkpoints/{best_or_latest}_checkpoint.pt'))
    if save_running:
        torch.save(pt, os.path.join(run_dir, f'checkpoints/checkpoint_{step}.pt'))


def load_checkpoint(run_dir, model, optimizer, scaler, best_or_latest):
    assert best_or_latest in ['best', 'latest'], 'best_or_latest must be "best" or "latest".'
    checkpoint = torch.load(os.path.join(run_dir, f'checkpoints/{best_or_latest}_checkpoint.pt'),
                            map_location=f'cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, scaler, checkpoint['early_stopping_score'], checkpoint['early_stopping_counter'], \
           checkpoint['running_mean'], checkpoint['running_std'], checkpoint['step']
