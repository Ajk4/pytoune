import warnings

import torch

from .lr_scheduler import PyTorchLRSchedulerWrapper, ReduceLROnPlateau
from .periodic import PeriodicSaveCallback


class ModelCheckpoint(PeriodicSaveCallback):
    """
    Save the model after every epoch. See
    `pytoune.framework.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        restore_best (bool): If `restore_best` is true, the weights of the
            network will be reset to the last best checkpoint done. This option
            only works when `save_best_only` is also true.
            (Default value = False)

    See:
        pytoune.framework.PeriodicSaveCallback
    """

    def __init__(self, *args, restore_best=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.restore_best = restore_best
        if self.restore_best and not self.save_best_only:
            raise ValueError("The 'restore_best' argument only works when "
                             "'save_best_only' is also true.")

    def save_file(self, fd, epoch, logs):
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
        }

        if self.model.lr_scheduler:
            state_dict['lr_scheduler_state_dict'] = self.model.lr_scheduler.state_dict(),

        torch.save(state_dict, fd)

    def on_train_end(self, logs):
        if self.restore_best:
            if self.best_filename is not None:
                if self.verbose:
                    print('Restoring model from %s' % self.best_filename)
                self.model.load_weights(self.best_filename)
            else:
                warnings.warn('No  weights to restore!')

    def load(self, pytoune_model):
        checkpoint = torch.load(self.filename)
        pytoune_model.model.load_state_dict(checkpoint['model_state_dict'])
        pytoune_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if pytoune_model.lr_scheduler:
            pytoune_model.lr_scheduler.load_state_dict('optimizer_state_dict')

        print(f"Loaded checkpoint {self.filename}")


class LRSchedulerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of an LR scheduler callback after every epoch. The LR
    scheduler callback should not be passed to the fit*() methods since it is
    called by this callback instead. The LR scheduler can be reloaded as
    follows.

    .. code-block:: python

        lr_scheduler = AnLRSchedulerCallback(...)
        lr_scheduler.load_state(filename)

    See `pytoune.framework.PeriodicSaveCallback` for the arguments'
    descriptions.

    Args:
        lr_scheduler: An LR scheduler callback.

    See:
        pytoune.framework.PeriodicSaveCallback
    """

    def __init__(self, lr_scheduler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = lr_scheduler

        if not isinstance(self.lr_scheduler, (PyTorchLRSchedulerWrapper, ReduceLROnPlateau)):
            raise ValueError("Unknown scheduler callback '%s'." % lr_scheduler)

    def save_file(self, fd, epoch, logs):
        self.lr_scheduler.save_state(fd)

    def set_params(self, params):
        self.lr_scheduler.set_params(params)
        super().set_params(params)

    def set_model(self, model):
        self.lr_scheduler.set_model(model)
        super().set_model(model)

    def on_epoch_begin(self, epoch, logs):
        self.lr_scheduler.on_epoch_begin(epoch, logs)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        self.lr_scheduler.on_epoch_end(epoch, logs)
        super().on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs):
        self.lr_scheduler.on_batch_begin(batch, logs)
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs):
        self.lr_scheduler.on_batch_end(batch, logs)
        super().on_batch_end(batch, logs)

    def on_backward_end(self, batch):
        self.lr_scheduler.on_backward_end(batch)
        super().on_backward_end(batch)

    def on_train_begin(self, logs):
        self.lr_scheduler.on_train_begin(logs)
        super().on_train_begin(logs)

    def on_train_end(self, logs):
        self.lr_scheduler.on_train_end(logs)
        super().on_train_end(logs)
