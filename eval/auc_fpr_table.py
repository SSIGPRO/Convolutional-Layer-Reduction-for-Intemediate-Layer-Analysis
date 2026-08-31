# python stuff
import re
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

import pandas as pd

# Our stuff
from configs.common import *

# maps the model macro of a row to the 'model' saved in the dataframes
model_macros = {
        'vgg': 'VGG',
        'mobilenet': 'MobileNet',
        'resnet': 'ResNet',
        'convnext': 'ConvNeXt',
        }

# maps the macro of a sub-row to the columns saved in the dataframes
split_macros = {
        'OOD': 'OoD',
        'AA': 'AA',
        'All': 'general',
        }

# columns without a reduction (the compared scores) are saved with '-'
default_reduction = '-'

# column order matching the table layout, left to right. Each cell holds both
# the AUC and the FPR of a split, printed as 'AUC\FPR'
col_order = []
for analysis in ['MACS', 'DMD']:
    for reduction in ['avgpooling', 'toeplitz', 'kernel']:
        col_order.append((analysis, reduction))
for score in ['MSP', 'DOC', 'Rel-U', 'FS']:
    col_order.append((score, default_reduction))

if __name__ == "__main__":
    dfs = {
            'aucs': pd.read_pickle(aucs_df_path),
            'fprs': pd.read_pickle(fprs_df_path),
            }

    metric_names = {'aucs': 'AUC', 'fprs': 'FPR'}

    # values[dataset][(model, split)][(analysis, reduction, metric)] = number
    values = {}
    for metric, df in dfs.items():
        for _, row in df.iterrows():
            for split in split_macros.values():
                col = f'{metric_names[metric]} {split}'
                if not col in row or pd.isna(row[col]): continue
                entry = values.setdefault(row['dataset'], {}).setdefault((row['model'], split), {})
                entry[(row['analysis'], row['reduction'], metric)] = row[col]

    template_lines = (results_path/'blank_auc_fpr_table.tex').read_text().splitlines()

    for dataset, ds_values in values.items():
        current_model = None
        new_lines = []
        for line in template_lines:
            if '\\caption{' in line:
                line = re.sub(r'\.\}', f' ({dataset}).}}', line)

            line = re.sub(r'\\label\{([^}]*)\}', lambda m: f'\\label{{{m.group(1)} {dataset}}}', line)

            mm = re.search(r'\\(%s)\b'%'|'.join(model_macros), line)
            if mm:
                current_model = model_macros[mm.group(1)]

            sm = re.search(r'\\aucGeom(\w+?)\\', line)
            split = split_macros[sm.group(1)] if sm and sm.group(1) in split_macros else None

            if current_model is not None and split is not None and '\\backslash' in line:
                entry = ds_values.get((current_model, split), {})

                new_vals = re.findall(r'\$([^$]*)\$', line)
                # the first cell is the row label, the values follow
                offset = len(new_vals) - len(col_order)
                for i, (analysis, reduction) in enumerate(col_order):
                    parts = new_vals[offset+i].split('\\backslash')
                    for p, metric in enumerate(['aucs', 'fprs']):
                        if (analysis, reduction, metric) in entry:
                            parts[p] = f'{entry[(analysis, reduction, metric)]:.2f}'
                    new_vals[offset+i] = '\\backslash'.join(parts)

                it = iter(new_vals)
                line = re.sub(r'\$([^$]*)\$', lambda m: f'${next(it)}$', line)

            new_lines.append(line)

        (results_path/f'aucsFprs_{dataset}.tex').write_text('\n'.join(new_lines) + '\n')
        print(f'Table saved in {results_path/f"aucsFprs_{dataset}.tex"}')
