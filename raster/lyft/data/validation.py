import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from os import listdir
from os.path import isfile, join
from collections import defaultdict
from .stats import traj_stat, classify_traj, comp_val
from raster.lyft.utils import find_batch_extremes, saliency_map, draw_batch
import torch
import tqdm
from captum.attr import Saliency
from pathlib import Path


def get_dataloader(data, tensor=False):
    res = defaultdict(list)
    for item in data:
        for key, value in item.items():
            res[key].append(value)
    for item, value in res.items():
        if tensor:
            res[item] = torch.Tensor(value)
        else:
            res[item] = np.array(value)
    return dict(**res)


class ValidateModel:
    def __init__(
            self, model, rasterizer, root: str = 'preprocess', grad_enabled: bool = True, device: str = 'cpu',
            turn_thresh: float = 3., speed_thresh: float = 0.5, k: int = 500, prog=True,
            output_root: str = 'validation', extreme_k: int = 5, visualize=True,

    ):
        self.root = root
        self.model = model.to(device)
        self.device = device
        self.grad_enabled = grad_enabled
        self.files = [f for f in listdir(root) if isfile(join(root, f)) and f.endswith('.npz')]
        self.splits = defaultdict(dict)
        self.k = k
        self.visualize = visualize
        self.turn_thresh = turn_thresh
        self.speed_thresh = speed_thresh
        self.prog = prog
        self.output_root = output_root
        Path(output_root).mkdir(parents=True, exist_ok=True)
        self.extreme_k = extreme_k
        self.rasterizer = rasterizer
        self.saliency = Saliency(self.model)

    def visualize_extremes(self, batch, res_type: str, kind: str, traj_cls: str):
        title = f'{traj_cls}-{res_type}-{kind}'
        res = draw_batch(self.rasterizer, batch, batch['outputs'])
        plt.rc('axes', titlesize=8)
        fig, axis = plt.subplots(1, res.shape[0], figsize=(50, 50))
        for idx, im in enumerate(res):
            (axis if len(res) == 1 else axis[idx]).imshow(im.transpose(1, 2, 0))
            if 'loss' in batch:
                (axis if len(res) == 1 else axis[idx]).set_title(f'loss: {batch["loss"][idx]}')
        fig.suptitle(f'raster: {title}', fontsize=16)
        fig.savefig(f'{self.output_root}/{title}-raster')
        plt.rc('axes', titlesize=4)
        fig, grad = saliency_map(batch, self.saliency, use_pyplot=False)
        fig.suptitle(f'saliency: {title}', fontsize=5)
        fig.savefig(f'{self.output_root}/{title}-saliency')

    def plot_loss_dist(self, df: pd.DataFrame):
        ax = sns.catplot(x='loss', y='type', hue='kind', data=df, kind='violin', dodge=True, height=25)
        loss_plot = sns.swarmplot(x='loss', y='type', hue='kind', color="k", data=df, ax=ax.ax, dodge=True, alpha=0.5)
        loss_plot.figure.savefig(f"{self.output_root}/plot-loss-dist")

    def plot_type_confusion(self, df: pd.DataFrame):
        types = np.unique(df['type'])
        cm = confusion_matrix(df['type'], df['pred_type'], labels=types)
        df_cm = pd.DataFrame(cm, index=[i for i in types],
                             columns=[i for i in types])
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1)
        cm_plot = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        cm_plot.figure.savefig(f"{self.output_root}/plot-type-confusion")

    def plot_scatter(self, df: pd.DataFrame, x_axis: str, y_axis: str):
        scatter = sns.FacetGrid(df, col="type", row="kind", height=5)
        scatter.map(sns.regplot, x=x_axis, y=y_axis, data=df, scatter_kws={'alpha': 0.1},
                    line_kws={'color': 'red'})
        scatter.add_legend()
        scatter.savefig(f"{self.output_root}/plot-scatter-{x_axis}-{y_axis}")

    def process_data(self, data, kind: str, traj_cls: str):
        batch = get_dataloader(data)
        images = torch.from_numpy(batch['image']).to(self.device)
        targets = torch.from_numpy(batch['target_positions']).to(self.device)
        with torch.set_grad_enabled(self.grad_enabled):
            loss, prediction = self.model(images, targets, return_outputs=True)
        res = defaultdict(list)
        if self.visualize:
            best_batch, worst_batch = find_batch_extremes(batch, loss, prediction, self.extreme_k, to_tensor=True)
            self.visualize_extremes(best_batch, 'best', kind, traj_cls)
            self.visualize_extremes(best_batch, 'worst', kind, traj_cls)

        for idx, item in enumerate(zip(prediction, loss)):
            pred, err = item
            traj = {
                'history_positions': batch['history_positions'][idx],
                'centroid': batch['centroid'][idx],
                'world_to_image': batch['world_to_image'][idx]
            }
            stats = traj_stat(traj, pred.detach().cpu().numpy())
            del traj
            pred_cls = classify_traj(
                *stats, turn_thresh=self.turn_thresh, speed_thresh=self.speed_thresh)
            pred_turn, pred_speed = comp_val(*stats, pred_cls)
            res['type'].append(traj_cls)
            res['pred_type'].append(pred_cls)
            res['kind'].append(kind)
            res['speed'].append(float(batch['speed'][idx]))
            res['turn'].append(float(batch['turn'][idx]))
            res['pred_turn'].append(float(pred_turn))
            res['pred_speed'].append(float(pred_speed))
            res['loss'].append(float(err.detach().cpu().numpy()))
        df = pd.DataFrame(res)
        del res, batch, images, targets, loss, prediction
        return df

    def validate_split(self, file: str):
        cls_name = file.replace('.npz', '')
        data = np.load(f'{self.root}/{file}', allow_pickle=True)['data'].item()
        extreme, sample = data['extreme'], data['sample']
        if len(extreme) < self.k:
            return self.process_data(extreme, 'extreme', cls_name)

        frames = [self.process_data(extreme, 'extreme', cls_name)]
        del extreme
        frames.append(self.process_data(sample, 'sample', cls_name))
        del sample
        return pd.concat(frames)

    def validate(self):
        frames = []
        prog = self.files if not self.prog else tqdm.tqdm(self.files, ascii=True)
        for file in prog:
            if self.prog:
                prog.set_postfix_str(file)
            frames.append(self.validate_split(file))
        df = pd.concat(frames)
        self.df = df
        self.plot_loss_dist(df)
        self.plot_scatter(df, 'loss', 'turn')
        self.plot_scatter(df, 'loss', 'speed')
        self.plot_scatter(df, 'pred_speed', 'speed')
        self.plot_scatter(df, 'pred_turn', 'turn')
        self.plot_type_confusion(df)

        return df
