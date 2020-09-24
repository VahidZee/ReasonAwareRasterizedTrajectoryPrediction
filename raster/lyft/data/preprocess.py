from collections import defaultdict, namedtuple
from typing import Any
import numpy as np
import pandas as pd
import bisect
import seaborn as sns
from pathlib import Path
from .stats import *
import tqdm

Item = namedtuple('Item', ['value', 'payload'])


class TrajSampler:
    k: int = 500
    p = 0.1

    def __init__(self):
        self.extreme = list()
        self.sample = list()
        self.size = 0

    def __len__(self):
        return min(len(self.extreme), len(self.sample))

    def add(self, trj_idx: int, value: Any):
        self.size += 1
        item = Item(value, trj_idx)
        bisect.insort(self.extreme, item)
        if len(self.extreme) > self.k:
            del self.extreme[0]
        if len(self.sample) < self.k:
            self.sample.append(item)
        elif np.random.rand(1) < self.p:
            self.sample.append(item)
            self.sample.pop(0)

    def stats_dict(self):
        res = defaultdict(list)
        for kind in ['sample', 'extreme']:
            for item in getattr(self, kind):
                res['turn'].append(item.value[0])
                res['speed'].append(item.value[1])
                res['payload'].append(item.payload)
                res['kind'].append(kind)
        return res

    def as_dict(self, dataset, kind=None):
        res = defaultdict(list)
        for que in ['extreme', 'sample']:
            for value, payload in getattr(self, que):
                rastered = dataset[payload]
                del rastered['target_availabilities']
                del rastered['history_availabilities']
                rastered['speed'] = value[1]
                rastered['turn'] = value[0]
                rastered['extreme'] = (que == 'extreme')
                if kind is not None:
                    rastered['kind'] = kind
                res[que].append(rastered)
        return res

    def __repr__(self):
        return f'(s:{len(self.sample)}, e:{len(self.extreme)})'


class DataSplitter:
    def __init__(self, dataset, prog=True,
                 k: int = 500, p: float = 0.1, turn_thresh=3., speed_thresh=0.5,
                 autosave=True, output_folder='preprocess', save_index=True):
        TrajSampler.k = k
        TrajSampler.p = p
        self.turn_thresh = turn_thresh
        self.speed_thresh = speed_thresh
        self.data = defaultdict(TrajSampler)
        self.prog = prog
        self.dataset = dataset
        self.autosave = autosave
        self.output_folder = output_folder
        self.size = 0
        self.save_index = save_index

    def stats_df(self):
        frames = []
        for traj_cls, traj_sampler in self.data.items():
            stats = traj_sampler.stats_dict()
            stats['type'] = [traj_cls] * (len(traj_sampler.sample) + len(traj_sampler.extreme))
            frames.append(pd.DataFrame(stats))
        return pd.concat(frames)

    def save(self, folder_name=None):
        folder_name = folder_name or self.output_folder
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        items = self.data.items()
        prog = items if not self.prog else tqdm.tqdm(items, ascii=True, desc='Saving')
        for name, samp in prog:
            if self.prog:
                prog.set_postfix_str(name)
            np.savez_compressed(f'{folder_name}/{name}', data=samp.as_dict(self.dataset))
        axes = self.plot_stats()
        axes[0].savefig(f'{folder_name}/turn')
        axes[1].savefig(f'{folder_name}/speed')

    def plot_stats(self, kind='strip', dodge=True, split=True, **kwargs):
        axes = []
        df = self.stats_df()
        for y in ['turn', 'speed']:
            if kind == 'violin':
                kwargs['inner'] = kwargs.get('innder', 'stick')
            ax = sns.catplot(
                x='type', y=y, hue='kind', data=df, kind=kind, dodge=dodge, split=split, **kwargs)

            ax.fig.suptitle(f'Capture of {len(self)} Samples')
            axes.append(ax)
        return axes

    @property
    def min_size(self):
        return min([len(s) for s in self.data.values()])

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"Capture of {len(self)} Samples" + "\n\t" + '\n\t'.join(
            f'{item}{samp}' for item, samp in self.data.items())

    def split(self, start=0, end=None, step=10):
        end = end or len(self.dataset)
        idxs = range(start, end, step)
        prog = idxs if not self.prog else tqdm.tqdm(idxs, ascii=True)
        try:
            for idx in prog:
                traj = self.dataset[idx]
                if filter_traj(traj):
                    continue
                stats = traj_stat(traj)
                traj_cls = classify_traj(*stats, self.turn_thresh, self.speed_thresh)
                val = comp_val(*stats, traj_cls)
                self.data[traj_cls].add(idx, val)
                self.size += 1
        finally:
            if self.save_index:
                self.stats_df.to_csv(f'{self.output_folder}/data_frame.csv')
            if self.autosave:
                self.save(self.output_folder)
