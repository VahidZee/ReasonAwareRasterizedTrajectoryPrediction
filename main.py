import pytorch_lightning as pl
from l5kit.configs import load_config_data
from raster.lyft import LyftTrainerModule, LyftDataModule
import argparse
import torch

parser = argparse.ArgumentParser(description='Manage running job')
parser.add_argument('--seed', type=int, default=313, help='random seed to use')
parser.add_argument('--config', type=str, required=True, help='config yaml path')
parser.add_argument('--log-lr', type=str, default='epoch', help='learning rate log interval')
parser.add_argument('--transfer', type=str, default=None, help='initial weights to transfer on')
parser = LyftTrainerModule.add_model_specific_args(parser)
parser = LyftDataModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    # initializing various parts
    pl.seed_everything(args.seed)

    # initializing training
    callbacks = []
    if args.log_lr:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval=args.log_lr))

    checkpoint = pl.callbacks.ModelCheckpoint(monitor='loss/val', save_last=True, verbose=True, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint, callbacks=callbacks)
    config = load_config_data(args.config)
    training_procedure = LyftTrainerModule(
        model=args.model, model_dict=args.model_dict, config=config, optimizer=args.optimizer,
        optimizer_dict=args.optimizer_dict, modes=args.modes, lr=args.lr, scheduler=args.scheduler,
        scheduler_dict=args.scheduler_dict, saliency_factor=args.saliency_factor,
        saliency_intrest=args.saliency_intrest, saliency_dict=args.saliency_dict, pgd_iters=args.pgd_iters,
        pgd_alpha=args.pgd_alpha, pgd_random_start=args.pgd_random_start, pgd_eps_vehicles=args.pgd_eps_vehicles,
        pgd_eps_semantics=args.pgd_eps_semantics, track_grad=args.track_grad)
    if args.transfer is not None:
        training_procedure.load_state_dict(torch.load(args.transfer)['state_dict'])
        print(args.transfer, 'successfully loaded as initial weights')

    training_procedure.datamodule = LyftDataModule(
        config=training_procedure.hparams.config, data_root=args.data_root, train_split=args.train_split,
        train_batch_size=args.train_batch_size, train_shuffle=args.train_shuffle,
        train_num_workers=args.train_num_workers, train_idxs=args.train_idxs, val_proportion=args.val_proportion,
        val_split=args.val_split, val_batch_size=args.val_batch_size, val_shuffle=args.val_shuffle,
        val_num_workers=args.val_num_workers, val_idxs=args.val_idxs)
    trainer.fit(training_procedure)
