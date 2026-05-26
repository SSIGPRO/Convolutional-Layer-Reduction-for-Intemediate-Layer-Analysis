# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

# plotting
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# torch
import torch

# Our stuff
from configs.common import *
from utils.get_best_configs import get_best_config

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def plot_isometrics(**kwargs):
    step = kwargs.get('step', 0.1)
    n_points = kwargs.get('n_points', 100)
    lb_x = kwargs.get('lb_x', 0)
    ub_x = kwargs.get('ub_x', 1)
    lb_y = kwargs.get('lb_y', 0)
    ub_y = kwargs.get('ub_y', 1)
    n_ood = kwargs.get('n_ood', 1)
    n_aa = kwargs.get('n_aa', 1)

    isos = torch.linspace(step, 1, int(1/step))
    for iso in isos:
        x = torch.linspace(1/n_points, 1, n_points)
        y = ((iso**(n_ood + n_aa))/(x**n_ood))**(1/n_aa)

        # filtering in range
        idx_x = torch.logical_and(x >= lb_x , x <= ub_x)
        idx_y = torch.logical_and(y >= lb_y , y <= ub_y)
        idx = torch.logical_and(idx_x, idx_y)
        x = x[idx]
        y = y[idx]
        if len(x) > 0 and len(y) > 0:
            plt.plot(x, y, c='xkcd:light grey', alpha=0.5, lw=0.2)
            plt.text(x[-1], y[-1], f'{iso:.2f}', c='xkcd:light grey', alpha=0.5, size='small')
    return


if __name__ == "__main__":
    hyperp_files = list(Path(args.data_path).glob('*/*/peepholes/*/*/hyperparams.pickle'))

    dfs = []
    best_configs = pd.DataFrame({})

    for hf in hyperp_files:
        dataset = hf.relative_to(args.data_path).parts[0]
        _df = pd.read_pickle(hf)[['AUC OoD', 'AUC AA', 'dataset', 'model', 'reduction', 'analysis']]
        dfs.append(_df)
        
        best_config = get_best_config(hf)[['AUC OoD', 'AUC AA', 'dataset', 'model', 'reduction', 'analysis']]
        best_configs = pd.concat([best_configs, best_config], ignore_index=True)

    # change the names to plot the bests according to the hue 
    best_configs = best_configs.replace({
        'avgpooling': 'best avgpooling',
        'toeplitz': 'best toeplitz',
        'kernel': 'best kernel'
        })

    df = pd.concat(dfs, ignore_index=True)
    df = pd.concat([df, best_configs], ignore_index=True)

    # latex like names
    auc_ood = r'$\overline{\Lambda}_{\rm{OoD}}$'
    auc_aa = r'$\overline{\Lambda}_{\rm{AA}}$'
    auc_all = r'$\overline{\Lambda}_{\rm{all}}$'

    df = df.rename(columns={'AUC OoD': auc_ood, 'AUC AA': auc_aa})

    results_path.mkdir(parents=True, exist_ok=True)

    _n_ood_per_ds = {'CIFAR100': 4, 'ImageNet': 3}

    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        n_ood = _n_ood_per_ds.get(dataset, len(ood_datasets))

        # plotting
        grid = sb.FacetGrid(
                data = df_ds,
                row = 'analysis',
                col = 'model',
                col_order = ['VGG', 'MobileNet', 'ResNet', 'ConvNeXt'],
                hue = 'reduction',
                hue_order = ['avgpooling', 'toeplitz', 'kernel','best avgpooling', 'best toeplitz', 'best kernel'],
                hue_kws = dict(
                    marker = ['.', '.', '.', 'd', 'd', 'd'],
                    ),
                palette = ['xkcd:ocean blue', 'xkcd:tangerine', 'xkcd:grass green', 'xkcd:electric blue', 'xkcd:bright orange', 'xkcd:forest']
                )

        grid.map(
                plot_isometrics,
                lb_x = df_ds[auc_ood].min(),
                ub_x = df_ds[auc_ood].max(),
                lb_y = df_ds[auc_aa].min(),
                ub_y = df_ds[auc_aa].max(),
                step = 0.05,
                n_ood = n_ood,
                n_aa = len(atk_names),
                )

        grid.map(
                sb.scatterplot,
                auc_ood,
                auc_aa,
                alpha = 0.6,
                s = 50,
                )

        # add the isometrics to the legend
        grid.set_titles('{col_name} | {row_name}')
        grid.add_legend()
        ax = plt.gca()
        leg = ax.legend(
                handles = [Line2D([0],[0], c='xkcd:light grey', lw=0.3, alpha=0.9, label=f'{auc_all}\nisometrics')],
                loc = 'upper left',
                bbox_to_anchor = (1.05, 0.6),
                frameon = False
                )
        ax.add_artist(leg)

        plt.savefig((results_path/f'aucs_{dataset}.png').as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
