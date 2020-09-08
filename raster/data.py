from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset

from typing import Union


class LyftDataModule(LightningDataModule):
    DEFAULT_BATCH_SIZE = 32

    def __init__(self, root_folder: str, config: Union[dict, str]):
        super().__init__()
        self.root_folder = root_folder
        self.config = config if isinstance(config, dict) else load_config_data(config)
        self.batch_size = min(
            self.config['train_data_loader'].get('batch_size', LyftDataModule.DEFAULT_BATCH_SIZE),
            self.config['val_data_loader'].get('batch_size', LyftDataModule.DEFAULT_BATCH_SIZE)
        )

    def setup(self, stage=None):
        self.data_manager = LocalDataManager(self.root_folder)
        self.rasterizer = build_rasterizer(self.config, self.data_manager)
        if stage == 'fit' or stage is None:
            train_zarr = ChunkedDataset(self.data_manager.require(self.config['train_data_loader']['key'])).open(
                cache_size_bytes=int(15e9)
            )
            self.train_data = AgentDataset(self.config, train_zarr, self.rasterizer)
            val_zarr = ChunkedDataset(self.data_manager.require(self.config['val_data_loader']['key'])).open(
                cache_size_bytes=int(5e9)
            )
            self.val_data = AgentDataset(self.config, val_zarr, self.rasterizer)
        if stage == 'test' or stage is None:
            pass  # no need to load the data separately

    def train_dataloader(self, batch_size=None):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.config['train_data_loader'].get('shuffle', True),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
