import os
import torch

from abc import ABC
from shutil import copyfile

from ray import tune

from ignite import utils
from ignite import engine
from ignite import distributed

from torch.utils import data
from torch.utils import tensorboard


class BaseModel(ABC):
    def __init__(self, job_dir, seed):
        super(BaseModel, self).__init__()
        self.job_dir = job_dir
        self.seed = seed

    def fit(self, train_dataset, tune_dataset):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement train()"
        )

    def save(self, is_best):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement save()"
        )

    def load(self, load_best):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement load()"
        )


class PyTorchModel(BaseModel):
    def __init__(self, job_dir, learning_rate, batch_size, epochs, num_workers, seed):
        super(PyTorchModel, self).__init__(job_dir=job_dir, seed=seed)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.summary_writer = tensorboard.SummaryWriter(log_dir=self.job_dir)
        self.logger = utils.setup_logger(
            name=__name__ + "." + self.__class__.__name__, distributed_rank=0
        )
        self.trainer = engine.Engine(self.train_step)
        self.evaluator = engine.Engine(self.tune_step)
        self._network = None
        self._optimizer = None
        self._metrics = None
        self.likelihood = None
        self.num_workers = num_workers
        self.device = distributed.device()

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def train_step(self, trainer, batch):
        raise NotImplementedError()

    def tune_step(self, trainer, batch):
        raise NotImplementedError()

    def on_epoch_completed(self, trainer, train_loader, tune_loader):
        raise NotImplementedError()

    def on_training_completed(self, trainer, loader):
        raise NotImplementedError()

    def preprocess(self, batch):
        inputs, targets = batch
        inputs = (
            [x.to(self.device) for x in inputs]
            if isinstance(inputs, list)
            else inputs.to(self.device)
        )
        targets = (
            [x.to(self.device) for x in targets]
            if isinstance(targets, list)
            else targets.to(self.device)
        )
        return inputs, targets

    def fit(self, train_dataset, tune_dataset):
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
        )
        tune_loader = data.DataLoader(
            tune_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        # Instantiate trainer
        for k, v in self.metrics.items():
            v.attach(self.trainer, k)
            v.attach(self.evaluator, k)
        self.trainer.add_event_handler(
            engine.Events.EPOCH_COMPLETED,
            self.on_epoch_completed,
            train_loader,
            tune_loader,
        )
        self.trainer.add_event_handler(
            engine.Events.COMPLETED, self.on_training_completed, tune_loader
        )
        self.load(load_best=False)
        # Train
        self.trainer.run(train_loader, max_epochs=self.epochs)
        return self.evaluator.state.metrics

    def on_epoch_completed(self, engine, train_loader, tune_loader):
        train_metrics = self.trainer.state.metrics
        print("Metrics Epoch", engine.state.epoch)
        justify = max(len(k) for k in train_metrics) + 2
        for k, v in train_metrics.items():
            if type(v) == float:
                print("train {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
            print("train {:<{justify}} {:<5}".format(k, v, justify=justify))
        self.evaluator.run(tune_loader)
        tune_metrics = self.evaluator.state.metrics
        if tune.is_session_enabled():
            tune.report(mean_loss=tune_metrics["loss"])
        justify = max(len(k) for k in tune_metrics) + 2
        for k, v in tune_metrics.items():
            if type(v) == float:
                print("tune {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
        is_best = tune_metrics["loss"] < self.best_loss
        self.best_loss = tune_metrics["loss"] if is_best else self.best_loss
        self.counter = 0 if is_best else self.counter + 1
        self.save(is_best=is_best)
        if self.counter == self.patience:
            self.logger.info(
                "Early Stopping: No improvement for {} epochs".format(self.patience)
            )
            engine.terminate()

    def on_training_completed(self, engine, loader):
        self.load(load_best=True)
        self.evaluator.run(loader)
        metric_values = self.evaluator.state.metrics
        print("Metrics Epoch", engine.state.epoch)
        justify = max(len(k) for k in metric_values) + 2
        for k, v in metric_values.items():
            if type(v) == float:
                print("best {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue

    def save(self, is_best):
        if not tune.is_session_enabled():
            state = {
                "model": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "engine": self.trainer.state,
            }
            if self.likelihood is not None:
                state["likelihood"] = self.likelihood.state_dict()
            p = os.path.join(self.job_dir, "checkpoint.pt")
            torch.save(state, p)
            if is_best:
                copyfile(p, os.path.join(self.job_dir, "best_checkpoint.pt"))

    def load(self, load_best=False):
        if tune.is_session_enabled():
            with tune.checkpoint_dir(step=self.trainer.state.epoch) as checkpoint_dir:
                p = os.path.join(checkpoint_dir, "checkpoint.pt")
        else:
            file_name = "best_checkpoint.pt" if load_best else "checkpoint.pt"
            p = os.path.join(self.job_dir, file_name)
        if not os.path.exists(p):
            self.logger.info(
                "Checkpoint {} does not exist, starting a new engine".format(p)
            )
            return
        self.logger.info("Loading saved checkpoint {}".format(p))
        checkpoint = torch.load(p)
        self.network.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.likelihood is not None:
            self.likelihood.load_state_dict(checkpoint["likelihood"])
        self.trainer.state = checkpoint["engine"]
