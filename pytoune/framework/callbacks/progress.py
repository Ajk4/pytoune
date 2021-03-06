import timeit
import sys

import itertools

from .callbacks import Callback


class ProgressionCallback(Callback):
    def on_train_begin(self, logs):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_train_end(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        self.step_times_sum = 0.
        self.epoch = epoch
        self.epoch_begin_time = timeit.default_timer()
        sys.stdout.write("\rEpoch %d/%d" % (self.epoch, self.epochs))
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs):
        self.epoch_end_time = timeit.default_timer()
        self.epoch_total_time = self.epoch_end_time - self.epoch_begin_time

        metrics_str = self._get_metrics_string(logs)
        if self.steps is not None:
            print("\rEpoch %d/%d %.2fs Step %d/%d: %s" % (
                self.epoch, self.epochs, self.epoch_total_time, self.steps, self.steps,
                metrics_str
            ))
        else:
            print("\rEpoch %d/%d %.2fs: Step %d/%d: %s" % (
                self.epoch, self.epochs, self.epoch_total_time, self.last_step, self.last_step,
                metrics_str
            ))

    def on_batch_begin(self, batch, logs):
        self.batch_begin_time = timeit.default_timer()

    def on_batch_end(self, batch, logs):
        self.batch_end_time = timeit.default_timer()
        self.step_times_sum += self.batch_end_time - self.batch_begin_time

        metrics_str = self._get_metrics_string(logs)

        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)

            sys.stdout.write("\rEpoch %d/%d ETA %.0fs Step %d/%d: %s" % (
                self.epoch, self.epochs, remaining_time, batch, self.steps, metrics_str
            ))
            sys.stdout.flush()
        else:
            sys.stdout.write("\rEpoch %d/%d %.2fs/step Step %d: %s" % (
                self.epoch, self.epochs, times_mean, batch, metrics_str
            ))
            sys.stdout.flush()
            self.last_step = batch

    def _get_metrics_string(self, logs):
        train_metrics_str = [f"loss: {'{:f}'.format(logs['loss'])}"]
        val_metrics_str = []

        if 'val_loss' in logs:
            val_metrics_str.append(f"val_loss: {'{:f}'.format(logs['val_loss'])}")

        for metric_name, metric_value in logs['metrics'].items():
            train_metrics_str.append(f"{metric_name}: {'{:f}'.format(metric_value)}")

        if 'val_metrics' in logs:
            for metric_name, metric_value in logs['val_metrics'].items():
                val_metrics_str.append(f"val_{metric_name}: {'{:f}'.format(metric_value)}")

        return ', '.join(itertools.chain(train_metrics_str, val_metrics_str))
