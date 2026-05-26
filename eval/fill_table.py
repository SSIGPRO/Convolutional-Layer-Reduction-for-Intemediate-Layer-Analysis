# python stuff
import re
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

# torch
import torch

# Our stuff
from configs.common import *
from utils.get_best_configs import get_best_config

#----------------------------------------------
# IMPORTANT: Script done with Claude Sonnet 4.6
# Revised and tested for correctness manually
#----------------------------------------------

# the hyperparams dataframe has columns named '{target_layer}/cv_dim',
# '{target_layer}/magnitude' or '{target_layer}/n_clusters'; recover the
# target layers (in their original, table-row order) from those columns.
def get_target_layers(bc):
    layers = []
    for col in bc.columns:
        m = re.match(r'(.+)/(?:cv_dim|magnitude|n_clusters|Q)$', col)
        if m and m.group(1) not in layers:
            layers.append(m.group(1))
    return layers

# maps a target layer name to the \layerName{...} label used in the table
def layer_label(layer):
    return layer.replace('features', 'f').replace('layer', 'l').replace('conv', 'c')

# maps the \multirow{...}{*}{\macro} model macro to the model name used above
model_macros = {
        'vgg': 'VGG',
        'mobilenet': 'MobileNet',
        'resnet': 'ResNet',
        'convnext': 'ConvNeXt',
        }

# column order matching the table layout, left to right
col_order = []
for analysis in ['DMD', 'MACS', 'MRC']:
    for reduction in ['avgpooling', 'toeplitz', 'kernel']:
        col_order.append((reduction, analysis, 'param'))
        if reduction != 'avgpooling':
            col_order.append((reduction, analysis, 'cv'))

if __name__ == "__main__":
    hyperp_files = list(Path(args.data_path).glob('*/*/peepholes/*/*/hyperparams.pickle'))

    # values[dataset][(model, layer_label)][f'{reduction}_{analysis}_{param|cv}'] = number
    values = {}

    for hf in hyperp_files:
        dataset, model, _, reduction, analysis, _ = hf.relative_to(args.data_path).parts

        bc = get_best_config(hf)

        for layer in get_target_layers(bc):
            entry = values.setdefault(dataset, {}).setdefault((model, layer_label(layer)), {})

            if analysis == 'DMD':
                param = int(bc[f'{layer}/magnitude'].iloc[0])/1000
            elif analysis == 'MRC':
                param = int(bc[f'{layer}/Q'].iloc[0])
            elif analysis == 'MACS':
                param = int(bc[f'{layer}/n_clusters'].iloc[0])
            else:
                raise RuntimeError('Unknown analysis. Check the analysis options.')
            entry[f'{reduction}_{analysis}_param'] = param

            if reduction != 'avgpooling':
                entry[f'{reduction}_{analysis}_cv'] = int(bc[f'{layer}/cv_dim'].iloc[0])

    template_lines = (results_path/'blank_table.tex').read_text().splitlines()

    for dataset, ds_values in values.items():
        current_model = None
        new_lines = []
        for line in template_lines:
            if '\\caption{' in line:
                line = re.sub(r'\.\}', f' ({dataset}).}}', line)

            mm = re.search(r'\\multirow\{\d+\}\{\*\}\{\\(\w+)\}', line)
            if mm and mm.group(1) in model_macros:
                current_model = model_macros[mm.group(1)]

            lm = re.search(r'\\layerName\{([^}]+)\}', line)
            if lm and current_model is not None:
                entry = ds_values.get((current_model, lm.group(1)))
                if entry is not None:
                    originals = re.findall(r'\$([^$]*)\$', line)
                    new_vals = list(originals)
                    for i, (reduction, analysis, kind) in enumerate(col_order):
                        key = f'{reduction}_{analysis}_{kind}'
                        if key in entry:
                            new_vals[i] = str(entry[key])

                    it = iter(new_vals)
                    line = re.sub(r'\$([^$]*)\$', lambda m: f'${next(it)}$', line)

            new_lines.append(line)

        (results_path/f'bestConfigs_{dataset}.tex').write_text('\n'.join(new_lines) + '\n')
