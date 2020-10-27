import pytorch_lightning as pl
from pytorch_lightning import loggers
from l5kit.configs import load_config_data
from raster.lyft import LyftTrainerModule, LyftDataModule
from pathlib import Path
import argparse
import torch
from raster.utils import boolify
import pandas as pd

parser = argparse.ArgumentParser(description='Manage running job')
parser.add_argument('--seed', type=int, default=313, help='random seed to use')
parser.add_argument('--config', type=str, help='config yaml path')
parser.add_argument('--checkpoint-path', type=str, default=None, help='initial weights to transfer on')
parser.add_argument('--challenge-submission', type=boolify, default=False,
                    help='whether test is for challenge submission')
parser.add_argument('--test-csv-path', type=str, default=None, help='where to save result of test')

parser = LyftTrainerModule.add_model_specific_args(parser)
parser = LyftDataModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    # initializing various parts
    pl.seed_everything(args.seed)

    # initializing training
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=False, logger=False)
    config = load_config_data(args.config)
    args_dict = vars(args)
    args_dict['config'] = config
    training_procedure = LyftTrainerModule.load_from_checkpoint( **args_dict)
    args_dict['config'] = training_procedure.hparams.config
    training_procedure.datamodule = LyftDataModule(**args_dict)
    trainer.test(training_procedure)
    if args_dict['challenge_submission']:
        validate_csv = pd.read_csv(args_dict['test_csv_path'] + "/full_result.csv")
        validate_csv.pop('idx')
        validate_csv.pop('grads/semantics')
        validate_csv.pop('grads/vehicles')
        validate_csv.pop('grads/total')
        validate_csv.pop('nll')
        validate_csv.pop('loss')
        validate_csv.to_csv(index=False, path_or_buf=args_dict['test_csv_path'] + "/submission.csv")
