import pytorch_lightning as pl
from l5kit.configs import load_config_data
from raster import BaseResnet, BaseTrainerModule, LyftDataModule
import os

if __name__ == '__main__':
    # initializing various parts
    config = load_config_data('./config.yaml')
    datamodule = LyftDataModule(config['train_params'].get('datapath', '~/lyft/'), config)
    training_procedure = BaseTrainerModule(config)
    training_procedure.datamodule = datamodule

    # initializing training
    trainer = pl.Trainer(**config['train_params'].get('trainer', dict()))
    trainer.fit(training_procedure)
