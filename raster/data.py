import torch
from pytorch_lightning import LightningDataModule

from l5kit.data import LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.dataset.dataloader_builder import build_dataloader
from l5kit.configs import load_config_data
from l5kit.dataset import AgentDataset

from typing import Union, Dict


class LyftDataModuel(LightningDataModule):
    def __init__(self, root_folder, config: Union[Dict, str], validation_portion=0.05):
        super().__init__()
        self.root_folder = root_folder
        self.validation_portion = validation_portion
        self.config = config if isinstance(config, dict) else load_config_data(config)
        self.data_manager = LocalDataManager(self.root_folder)
        self.rasterizer = build_rasterizer(self.config, self.data_manager)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        t_data = build_dataloader(self.config, "train", self.data_manager, AgentDataset,
                                  self.rasterizer).dataset
        train_len = int(len(t_data) * (1 - self.validation_portion))
        self.train_data, self.val_data = torch.utils.data.random_split(t_data, [train_len, len(t_data) - train_len])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.config['train_data_loader']['batch_size'],
            shuffle=self.config['train_data_loader'].get('shuffle', True),
        )

    def val_data_loader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.config['train_data_loader']['batch_size'],
            shuffle=False,
        )

    def test_dataloader(self):
        return build_dataloader(self.config, "val", self.data_manager, AgentDataset,
                                self.rasterizer)
