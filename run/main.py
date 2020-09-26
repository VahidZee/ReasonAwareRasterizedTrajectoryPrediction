import pytorch_lightning as pl
from l5kit.configs import load_config_data
from raster.lyft import LyftTrainerModule, LyftDataModule
import argparse

parser = argparse.ArgumentParser(description='Manage running job')
parser.add_argument('--dataset', choices=('lyft',), default='lyft', help='desired dataset to be used')
parser.add_argument('--config', default='./config.yaml', help='path to config file')
parser.add_argument('--data_root', default='~/lyft/', help='path to dataset folder')
parser = LyftTrainerModule.add_model_specific_args(parser)

parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

if __name__ == '__main__':
    # initializing various parts
    config = load_config_data('../config.yaml')
    datamodule = LyftDataModule(config['train_params'].get('datapath', '~/lyft/'), config)
    training_procedure = LyftTrainerModule(config)
    training_procedure.datamodule = datamodule

    # initializing training
    trainer = pl.Trainer.from_argparse_args(args, **config.get('train_params', dict()).get('trainer', dict()))
    trainer.fit(training_procedure)
