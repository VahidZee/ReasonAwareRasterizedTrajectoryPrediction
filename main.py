import pytorch_lightning as pl
from l5kit.configs import load_config_data
from raster import BaseResnet, BaseTrainerModule, LyftDataModule
import os

if __name__ == '__main__':
    # initializing various parts
    config = load_config_data(os.environ.get('config_path'))
    datamodule = LyftDataModule(os.environ.get('dataset_path'), config)
    training_procedure = BaseTrainerModule(config)
    training_procedure.datamodule = datamodule

    # initializing training
    trainer = pl.Trainer(gpus=os.environ.get('gpu_count'), max_epochs=5)
    trainer.fit(training_procedure)
