import csv
from .callbacks import Callback

class Logger(Callback):
    def __init__(self, *, batch_granularity=False):
        super().__init__()
        self.batch_granularity = batch_granularity
        self.epoch = 0

    def on_train_begin(self, logs):
        self._on_train_begin_write(logs)

    def _on_train_begin_write(self, logs):
        pass

    def on_batch_end(self, batch, logs):
        if self.batch_granularity:
            self._on_batch_end_write(batch, logs)

    def _on_batch_end_write(self, batch, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch
        self._on_epoch_begin_write(epoch, logs)

    def _on_epoch_begin_write(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        self._on_epoch_end_write(epoch, logs)

    def _on_epoch_end_write(self, epoch, logs):
        pass

    def on_train_end(self, logs=None):
        self._on_train_end_write(logs)

    def _on_train_end_write(self, logs):
        pass

    def _get_current_learning_rates(self):
        learning_rates = [param_group['lr'] for param_group in self.model.optimizer.param_groups]
        return learning_rates[0] if len(learning_rates) == 1 else learning_rates



class TensorBoardLogger(Logger):
    """
    Callback that output the result of each epoch or batch into a Tensorboard experiment folder.

    Args:
        writer (tensorboardX.SummaryWriter): The tensorboard writer.

    Example:
        Using tensorboardX::

            from tensorboardX import SummaryWriter
            from pytoune.framework import Model
            from pytoune.framework.callbacks import TensorBoardLogger

            writer = SummaryWriter('runs')
            tb_logger = TensorBoardLogger(writer)

            model = Model(...)
            model.fit_generator(..., callbacks=[tb_logger])
    """
    def __init__(self, writer):
        super().__init__(batch_granularity=False)

        self.writer = writer

    def _on_batch_end_write(self, batch, logs):
        """
        We don't handle tensorboard writing on batch granularity
        """
        pass

    def _on_epoch_end_write(self, epoch, logs):
        lr = self._get_current_learning_rates()

        if isinstance(lr, (list,)):
            lr_scalars = {'lr_' + str(i): v for i, v in enumerate(lr)}
        else:
            lr_scalars = {'lr': lr}

        for metric_name, metric_value in lr_scalars.items():
            self.writer.add_scalar(f'lr/${metric_name}', metric_value, epoch)

        self.writer.add_scalar(f'train/loss', logs['loss'], epoch)
        for metric_name, metric_value in logs['metrics'].items():
            self.writer.add_scalar(f'train/${metric_name}', metric_value, epoch)

        if 'val_loss' in logs:
            self.writer.add_scalar('val/loss', logs['val_loss'], epoch)
            for metric_name, metric_value in logs['val_metrics'].items():
                self.writer.add_scalar(f'val/${metric_name}', metric_value, epoch)

        self.writer.flush()
