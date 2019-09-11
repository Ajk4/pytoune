import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytoune import torch_to_numpy, numpy_to_torch, torch_to
from .iterators import EpochIterator, StepIterator, _get_step_iterator
from .callbacks import CallbackList, ProgressionCallback, Callback
from .metrics import get_loss_or_metric
from .optimizers import get_optimizer
from .warning_manager import warning_settings


class Model:
    """
    The Model class encapsulates a PyTorch module/network, a PyTorch optimizer,
    a loss function and metric functions. It allows the user to train a neural
    network without hand-coding the epoch/step logic.

    Args:
        model (torch.nn.Module): A PyTorch module.
        optimizer (torch.optim.Optimizer): Initialized PyTorch optimizer.

    Attributes:
        model (torch.nn.Module): The associated PyTorch module.
        optimizer (torch.optim.Optimizer): The associated PyTorch optimizer.

    Example:
        Using Numpy arrays (or tensors) dataset::

            from pytoune.framework import Model
            import torch
            import numpy as np

            num_features = 20
            num_classes = 5

            # Our training dataset with 800 samples.
            num_train_samples = 800
            train_x = np.random.randn(num_train_samples, num_features).astype('float32')
            train_y = np.random.randint(num_classes, size=num_train_samples)

            # Our validation dataset with 200 samples.
            num_valid_samples = 200
            valid_x = np.random.randn(num_valid_samples, num_features).astype('float32')
            valid_y = np.random.randint(num_classes, size=num_valid_samples)

            pytorch_module = torch.nn.Linear(num_features, num_classes) # Our network

            # We create and optimize our model
            model = Model(pytorch_module, 'sgd', 'cross_entropy', metrics=['accuracy'])
            model.fit(train_x, train_y,
                      validation_x=valid_x,
                      validation_y=valid_y,
                      epochs=5,
                      batch_size=32)

        .. code-block:: none

            Epoch 1/10 0.01s Step 40/40: loss: 0.710869, val_loss: 0.489602
            Epoch 2/10 0.01s Step 40/40: loss: 0.448081, val_loss: 0.305897
            Epoch 3/10 0.01s Step 40/40: loss: 0.301377, val_loss: 0.204526
            ...

        Using PyTorch DataLoader::

           import torch
           from torch.utils.data import DataLoader, TensorDataset
           from pytoune.framework import Model

           num_features = 20
           num_classes = 5

           # Our training dataset with 800 samples.
           num_train_samples = 800
           train_x = torch.rand(num_train_samples, num_features)
           train_y = torch.randint(num_classes, (num_train_samples,), dtype=torch.long)
           train_dataset = TensorDataset(train_x, train_y)
           train_generator = DataLoader(train_dataset, batch_size=32)

           # Our validation dataset with 200 samples.
           num_valid_samples = 200
           valid_x = torch.rand(num_valid_samples, num_features)
           valid_y = torch.randint(num_classes, (num_valid_samples,), dtype=torch.long)
           valid_dataset = TensorDataset(valid_x, valid_y)
           valid_generator = DataLoader(valid_dataset, batch_size=32)

           pytorch_module = torch.nn.Linear(num_features, num_train_samples)

           model = Model(pytorch_module, 'sgd', 'cross_entropy', metrics=['accuracy'])
           model.fit_generator(train_generator,
                               valid_generator,
                               epochs=5)

        .. code-block:: none

            Epoch 1/10 0.01s Step 40/40: loss: 0.311442, val_loss: 0.243208
            Epoch 2/10 0.01s Step 40/40: loss: 0.223419, val_loss: 0.183428
            Epoch 3/10 0.01s Step 40/40: loss: 0.173739, val_loss: 0.150269
            ...

    """

    def __init__(self, model, optimizer, *, lr_scheduler=None, clip_grad_value=None, opt_iter=1):
        self.model = model
        self.optimizer = get_optimizer(optimizer, self.model)
        self.lr_scheduler = lr_scheduler
        self.clip_grad_value = clip_grad_value
        self.device = None
        self.opt_iter = opt_iter

    def fit(self, x, y, validation_x=None, validation_y=None, *, shuffle=False,
            batch_size=32, epochs=1000, steps_per_epoch=None, validation_steps=None,
            initial_epoch=1, verbose=True, callbacks=[]):
        """
        Trains the model on a dataset. This method creates generators and calls
        the ``fit_generator`` method.

        Args:
            x (Union[Tensor, np.ndarray]): Training dataset.
            y (Union[Tensor, np.ndarray]): Ground truth.
            validation_x (Union[Tensor, np.ndarray]): Validation dataset. The validation datset
                is optional. (Default value = None)
            validation_y (Union[Tensor, np.ndarray]): Validation ground truth.
                (Default value = None)
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one
                epoch. Obviously, using this argument may cause one epoch not to
                see the entire training dataset or see it multiple times.
                (Defaults the number of steps needed to see the entire
                training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch`` but
                for the validation dataset. (Defaults to ``steps_per_epoch`` if
                provided or the number of steps needed to see the entire
                validation dataset)
            initial_epoch (int, optional): Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            callbacks (list of pytoune.framework.Callback): List of callbacks
                that will be called during training. (Default value = [])

        Returns:
            List of dict containing the history of each epoch.

        Example:
            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function)
                history = model.fit(train_x, train_y,
                                    validation_x=valid_x,
                                    validation_y=valid_y,
                                    epochs=num_epochs,
                                    batch_size=batch_size)
                                    verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 0.30211143642663957, 'val_loss': 0.25165273696184159}
                {'epoch': 2, 'loss': 0.2192931968718767, 'val_loss': 0.19234802126884459}
                {'epoch': 3, 'loss': 0.17256419658660888, 'val_loss': 0.15897458493709565}
                ...

        """

        if isinstance(x, list):
            raise Exception("fit() does not support multiple inputs. Use fit_generator")

        train_generator = self._dataloader_from_data(x, y, batch_size=batch_size, shuffle=shuffle)
        valid_generator = None
        if validation_x is not None or validation_y is not None:
            valid_generator = self._dataloader_from_data(validation_x,
                                                         validation_y,
                                                         batch_size=batch_size)

        return self.fit_generator(train_generator,
                                  valid_generator=valid_generator,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  initial_epoch=initial_epoch,
                                  verbose=verbose,
                                  callbacks=callbacks)

    def _dataloader_from_data(self, *args, batch_size=None, shuffle=False):
        assert batch_size is not None, \
            "batch_size should not be None. Please, report this as a bug."
        args = numpy_to_torch(args)
        dataset = TensorDataset(*args) if len(args) > 1 else args[0]
        generator = DataLoader(dataset, batch_size, shuffle=shuffle)
        return generator

    def fit_generator(self, train_generator, valid_generator=None, *,
                      epochs=1000, steps_per_epoch=None, validation_steps=None,
                      initial_epoch=1, verbose=True, callbacks=[]):
        # pylint: disable=too-many-locals
        """
        Trains the model on a dataset using a generator.

        Args:
            train_generator: Generator-like object for the training dataset.
                The generator must yield a tuple ``(x, y)`` where ``x`` is a
                batch of the training dataset and ``y`` is the corresponding
                ground truths. ``y`` should be a Tensor or a Numpy array with
                the first dimension being the batch size since ``len(y)`` is
                taken as the batch size. The loss and the metrics are averaged
                using this batch size. If ``y`` is not a Tensor or a Numpy
                array, then a warning is raised and the "batch size" defaults
                to 1.

                If the generator does not have a method ``__len__()``, either
                the ``steps_per_epoch`` argument must be provided, or the
                iterator returned raises a StopIteration exception at the end
                of the training dataset. PyTorch DataLoaders object do provide a
                ``__len__()`` method.

                Before each epoch, the method ``__iter__()`` on the generator is
                called and the method ``__next__()`` is called for each step on
                resulting object returned by ``__iter__()``. Notice that a call
                to ``__iter__()`` on a generator made using the python keyword
                ``yield`` returns the generator itself.
            valid_generator (optional): Generator-like object for the
                validation dataset. This generator is optional. The generator is
                used the same way as the  generator ``train_generator``. If the
                generator does not have a method ``__len__()``, either the
                ``validation_steps`` or the ``steps_per_epoch`` argument must be
                provided or the iterator returned raises a StopIteration
                exception at the end of the validation dataset.
                (Default value = None)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one
                epoch. Obviously, using this argument may cause one epoch not to
                see the entire training dataset or see it multiple times.
                (Defaults the number of steps needed to see the entire
                training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch``
                but for the validation dataset. (Defaults to ``steps_per_epoch``
                if provided or the number of steps needed to see the entire
                validation dataset)
            initial_epoch (int, optional): Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            callbacks (list of pytoune.framework.Callback): List of callbacks
                that will be called during training. (Default value = [])

        Returns:
            List of dict containing the history of each epoch.

        Example:
            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function)
                history = model.fit_generator(train_generator,
                                              valid_generator,
                                              epochs=num_epochs,
                                              verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 0.4048105351626873, 'val_loss': 0.35831213593482969}
                {'epoch': 2, 'loss': 0.27947457544505594, 'val_loss': 0.25963697880506514}
                {'epoch': 3, 'loss': 0.20913131050765515, 'val_loss': 0.20263003259897233}
                ...

        """
        self._transfer_optimizer_state_to_right_device()

        if verbose:
            callbacks = [ProgressionCallback()] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        self.optimizer.zero_grad()

        self.stop_training = False
        epoch_iterator = EpochIterator(train_generator, valid_generator,
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_steps=validation_steps,
                                       initial_epoch=initial_epoch,
                                       callback=callback_list)

        for train_step_iterator, valid_step_iterator in epoch_iterator:
            self.model.train(True)
            with torch.enable_grad():
                for step, inputs in train_step_iterator:
                    step.loss, step.metrics = self._fit_batch(inputs,
                                                              callback=callback_list,
                                                              step=step.number)
                    step.size = self._get_batch_size(inputs)

            if valid_step_iterator is not None:
                self._validate(valid_step_iterator)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            epoch_iterator.stop_training = self.stop_training

        return epoch_iterator.epoch_logs

    def _fit_batch(self, inputs, *, callback=CallbackList(), step):
        loss_tensor, metrics = self.model(*inputs)

        loss_tensor = loss_tensor / self.opt_iter

        loss_tensor.backward()
        
        callback.on_backward_end(step)

        if self.clip_grad_value:
            # Ideally parameters from optimizer should be used
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_grad_value)

        if step % self.opt_iter == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.optimizer is not None and hasattr(self.optimizer, 'batch_step'):
            self.optimizer.batch_step()

        loss = float(loss_tensor)
        return loss, metrics

    def _process_input(self, *args):
        args = numpy_to_torch(args)
        if self.device is not None:
            args = torch_to(args, self.device)
        return args[0] if len(args) == 1 else args

    def train_on_batch(self, inputs):
        """
        Trains the model for the batch ``(x, y)`` and computes the loss and
        the metrics, and optionaly returns the predictions.

        Args:
            x: Batch.
            y: Batch ground truths.

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is
            false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the predictions with tensors converted into Numpy
            arrays.
        """
        self.model.train(True)
        with torch.enable_grad():
            self._transfer_optimizer_state_to_right_device()
            loss, metrics = self._fit_batch(inputs)
        return self._format_return(loss, metrics)

    def _format_return(self, loss, metrics):
        ret = (loss,)

        # ret += tuple(metrics.tolist()) if len(metrics) <= 1 else (metrics,)

        return ret[0] if len(ret) == 1 else ret

    def predict(self, x, *, batch_size=32):
        """
        Returns the predictions of the network given a dataset ``x``, where the
        tensors are converted into Numpy arrays.

        Args:
            x (Union[Tensor, np.ndarray]): Dataset for which to predict.
            batch_size (int): Number of samples given to the network at one
                time. (Default value = 32)

        Returns:
            Numpy arrays of the predictions.
        """
        generator = self._dataloader_from_data(x, batch_size=batch_size)
        pred_y = self.predict_generator(generator)
        return np.concatenate(pred_y)

    def predict_generator(self, generator, *, steps=None):
        """
        Returns the predictions of the network given batches of samples ``x``,
        where the tensors are converted into Numpy arrays.

        generator: Generator-like object for the dataset. The generator must
            yield a batch of samples. See the ``fit_generator()`` method for
            details on the types of generators supported.
        steps (int, optional): Number of iterations done on
            ``generator``. (Defaults the number of steps needed to see the
            entire dataset)

        Returns:
            List of the predictions of each batch with tensors converted into
            Numpy arrays.
        """
        if steps is None and hasattr(generator, '__len__'):
            steps = len(generator)
        pred_y = []
        self.model.eval()
        with torch.no_grad():
            for _, x in _get_step_iterator(steps, generator):
                x = self._process_input(x)
                pred_y.append(torch_to_numpy(self.model(x)))
        return pred_y

    def predict_on_batch(self, x):
        """
        Returns the predictions of the network given a batch ``x``, where the
        tensors are converted into Numpy arrays.

        Args:
            x (Union[Tensor, np.ndarray]): Batch for which to predict.

        Returns:
            The predictions with tensors converted into Numpy arrays.
        """
        self.model.eval()
        with torch.no_grad():
            x = self._process_input(x)
            return torch_to_numpy(self.model(x))

    def evaluate(self, x, y, *, batch_size=32):
        """
        Computes the loss and the metrics of the network on batches of samples
        and optionaly returns the predictions.

        Args:
            x (Union[Tensor, np.ndarray]): Dataset.
            y (Union[Tensor, np.ndarray]): Dataset ground truths.
            batch_size (int): Number of samples given to the network at one
                time. (Default value = 32)
        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is
            false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is a Numpy array of the predictions.
        """
        generator = self._dataloader_from_data(x, y, batch_size=batch_size)
        ret = self.evaluate_generator(generator, steps=len(generator))
        return ret


    def evaluate_generator(self, generator, *, steps=None):
        """
        Computes the loss and the metrics of the network on batches of samples
        and optionaly returns the predictions.

        Args:
            generator: Generator-like object for the dataset. The generator
                must yield a tuple ``(x, y)`` where ``x`` is a batch of the
                dataset and ``y`` is the corresponding ground truths. ``y``
                should be a Tensor or a Numpy array with the first dimension
                being the batch size since ``len(y)`` is taken as the batch
                size. The loss and the metrics are averaged using this batch
                size. If ``y`` is not a Tensor or a Numpy array, then a warning
                is raised and the "batch size" defaults to 1.

                See the ``fit_generator()`` method for details on the types of
                generators supported.
            steps (int, optional): Number of iterations done on
                ``generator``. (Defaults the number of steps needed to see the
                entire dataset)

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is
            false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the list of the predictions of each batch with tensors
            converted into Numpy arrays.

        Example:
            With no metrics:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=[])
                loss = model.evaluate_generator(test_generator)

            With only one metric:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=[my_metric_fn])
                loss, my_metric = model.evaluate_generator(test_generator)

            With only several metrics:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=[my_metric1_fn, my_metric2_fn])
                loss, (my_metric1, my_metric2) = model.evaluate_generator(test_generator)

            With metrics and ``return_pred`` flag:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=[my_metric1_fn, my_metric2_fn])
                loss, (my_metric1, my_metric2), pred_y = model.evaluate_generator(
                    test_generator, return_pred=True
                )
        """
        if steps is None:
            steps = len(generator)
        step_iterator = StepIterator(generator, steps, Callback())
        loss, metrics = self._validate(step_iterator)
        return loss, metrics

    def evaluate_on_batch(self, inputs):
        """
        Computes the loss and the metrics of the network on a single batch of
        samples and optionaly returns the predictions.

        Args:
            x (Union[Tensor, np.ndarray]): Batch.
            y (Union[Tensor, np.ndarray]): Batch ground truths.

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is
            false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the predictions with tensors converted into Numpy
            arrays.
        """
        self.model.eval()
        with torch.no_grad():
            loss, metrics = self.model(*inputs)
        return self._format_return(loss, metrics)

    def _validate(self, step_iterator):
        self.model.eval()
        with torch.no_grad():
            for step, inputs in step_iterator:
                step.loss, step.metrics = self.model(*inputs)
                step.size = self._get_batch_size(inputs)

        return step_iterator.loss, step_iterator.metrics

    def _get_batch_size(self, inputs):
        x = inputs[0]
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            return len(x)
        elif warning_settings['batch_size'] == 'warn':
            warnings.warn("When 'x' or 'y' are not tensors nor Numpy arrays, "
                          "the batch size is set to 1 and, thus, the computed "
                          "loss and metrics at the end of each epoch is the "
                          "mean of the batches' losses and metrics. To disable "
                          "this warning, set\n"
                          "from pytoune.framework import import warning_settings\n"
                          "warning_settings['batch_size'] = 'ignore'")
        return 1

    def load_weights(self, f):
        """
        Loads the weights saved using the ``torch.save()`` method or the
        ``save_weights()`` method of this class. Contrary to ``torch.load()``,
        the weights are not transfered to the device from which they were saved
        from. In other words, the PyTorch module will stay on the same device it
        already is on.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        self.set_weights(torch.load(f, map_location='cpu'))

    def save_weights(self, f):
        """
        Saves the weights of the current network.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        torch.save(self.model.state_dict(), f)

    def load_optimizer_state(self, f):
        """
        Loads the optimizer state saved using the ``torch.save()`` method or the
        ``save_optimizer_state()`` method of this class.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        self.optimizer.load_state_dict(torch.load(f, map_location='cpu'))

    def save_optimizer_state(self, f):
        """
        Saves the state of the current optimizer.

        Args:
            f: File-like object (has to implement fileno that returns a file
                descriptor) or string containing a file name.
        """
        torch.save(self.optimizer.state_dict(), f)

    def _transfer_optimizer_state_to_right_device(self):
        # Since the optimizer state is loaded on CPU, it will crashed when the
        # optimizer will receive gradient for parameters not on CPU. Thus, for
        # each parameter, we transfer its state in the optimizer on the same
        # device as the parameter itself just before starting the optimization.
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    for _, v in self.optimizer.state[p].items():
                        if torch.is_tensor(v) and p.device != v.device:
                            v.data = v.data.to(p.device)

    def get_weights(self):
        """
        Returns a dictionary containing the parameters of the network. The
        tensors are just references to the parameters. To get copies of the
        weights, see the ``get_weight_copies()`` method.
        """
        return self.model.state_dict()

    def get_weight_copies(self):
        """
        Returns a dictionary containing copies of the parameters of the network.
        """
        weights = self.get_weights()
        for k in weights.keys():
            weights[k] = weights[k].cpu().clone()
        return weights

    def set_weights(self, weights):
        """
        Modifies the weights of the network with the given weights.

        Args:
            weights (dict): Weights returned by either ``get_weights()`` or
                ``get_weight_copies()``.
        """
        self.model.load_state_dict(weights)

    def cuda(self, *args, **kwargs):
        """
        Tranfers the network on the GPU. The arguments are passed to the
        ``torch.nn.Module.cuda()`` method. Notice that the device is saved so
        that the batches can send to the right device before passing it to the
        network.

        Returns:
            `self`.
        """
        self.model.cuda(*args, **kwargs)
        self.device = None
        for _, p in zip(range(1), self.model.parameters()):
            self.device = p.device
        return self

    def cpu(self, *args, **kwargs):
        """
        Tranfers the network on the CPU. The arguments are passed to the
        ``torch.nn.Module.cpu()`` method. Notice that the device is saved so
        that the batches can send to the right device before passing it to the
        network.

        Returns:
            `self`.
        """
        self.model.cpu(*args, **kwargs)
        self.device = None
        for _, p in zip(range(1), self.model.parameters()):
            self.device = p.device
        return self

    def to(self, device):
        """
        Tranfers the network on the specified device. The device is saved so
        that the batches can send to the right device before passing it to the
        network.

        Args:
            device (torch.device): The device to which the network is sent.

        Returns:
            `self`.
        """
        self.device = device
        self.model.to(self.device)
        return self
