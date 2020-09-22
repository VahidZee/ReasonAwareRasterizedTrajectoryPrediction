from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule

from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset

from typing import Union


class LyftDataModule(LightningDataModule):
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_CACHE_SIZE = int(1e9)
    DEFAULT_NUM_WORKERS = 8

    def __init__(self, root_folder: str, config: Union[dict, str]):
        super().__init__()
        self.root_folder = root_folder
        self.config = config if isinstance(config, dict) else load_config_data(config)
        self.batch_size = min(
            self.config.get('train_dataloader', dict()).get('batch_size', LyftDataModule.DEFAULT_BATCH_SIZE),
            self.config.get('val_dataloader', dict()).get('batch_size', LyftDataModule.DEFAULT_BATCH_SIZE),
            self.config.get('test_dataloader', dict()).get('batch_size', LyftDataModule.DEFAULT_BATCH_SIZE),
        )

    def setup(self, stage=None):
        self.data_manager = LocalDataManager(self.root_folder)
        self.rasterizer = build_rasterizer(self.config, self.data_manager)
        if stage == 'fit' or stage is None:
            print(self.config['train_dataloader'].get('splits',
                                                      [{'key': self.config['train_dataloader'].get('key', None)}]))
            train_zarrs = [
                ChunkedDataset(self.data_manager.require(split['key'])).open(
                    cache_size_bytes=int(split.get('cache', self.DEFAULT_CACHE_SIZE))) for split in
                self.config['train_dataloader'].get('splits',
                                                    [{'key': self.config['train_dataloader'].get('key', None)}])]
            self.train_data = ConcatDataset([AgentDataset(self.config, zarr, self.rasterizer) for zarr in train_zarrs])
            val_zarrs = [
                ChunkedDataset(self.data_manager.require(split['key'])).open(
                    cache_size_bytes=int(split.get('cache', self.DEFAULT_CACHE_SIZE))) for split in
                self.config['train_dataloader'].get('splits',
                                                    [{'key': self.config['val_dataloader'].get('key', None)}])]
            self.val_data = ConcatDataset([AgentDataset(self.config, zarr, self.rasterizer) for zarr in val_zarrs])
        if (stage == 'test' or stage is None) and 'test_dataloader' in self.config:
            test_zarrs = [
                ChunkedDataset(self.data_manager.require(split['key'])).open(
                    cache_size_bytes=int(split.get('cache', self.DEFAULT_CACHE_SIZE))) for split in
                self.config['test_dataloader'].get('splits', [{'key': self.config['val_dataloader'].get('key', None)}])]
            self.test_data = ConcatDataset([AgentDataset(self.config, zarr, self.rasterizer) for zarr in test_zarrs])

    def train_dataloader(self, batch_size=None, num_workers=None):
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.config['val_dataloader'].get('num_workers', self.DEFAULT_NUM_WORKERS)
        return DataLoader(
            self.train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=self.config['train_dataloader'].get('shuffle', True),
        )

    def val_dataloader(self, batch_size=None, num_workers=None):
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.config['val_dataloader'].get('num_workers', self.DEFAULT_NUM_WORKERS)
        return DataLoader(
            self.val_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=self.config['val_dataloader'].get('shuffle', True),
        )

    def test_dataloader(self, batch_size=None, num_workers=None):
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.config['test_dataloader'].get('num_workers', self.DEFAULT_NUM_WORKERS)
        return DataLoader(
            self.test_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=self.config['test_dataloader'].get('shuffle', True),
        )
