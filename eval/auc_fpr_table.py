# python stuff
import re
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

import pandas as pd

# Our stuff
from configs.common import *

# maps the \gls{...} or \<macro> of a row to the 'analysis' saved in the dataframes
analysis_macros = {
        'gls{macs}': 'MACS',
        'gls{dmd}': 'DMD',
        'mspScore': 'MSP',
        'doctorScore': 'DOC',
        'reluScore': 'Rel-U',
        'fsScore': 'FS',
        }

# rows without a reduction sub-row (the compared scores) are saved with '-'
default_reduction = '-'

# maps the reduction macro of a sub-row to the 'reduction' saved in the dataframes
reduction_macros = {
        'avgpDimRed': 'avgpooling',
        'toeplitzDimRed': 'toeplitz',
        'kernelDimRed': 'kernel',
        }

# column order matching the table layout, left to right
col_order = []
for model in ['VGG', 'MobileNet', 'ResNet', 'ConvNeXt']:
    for split in ['OoD', 'AA', 'general']:
        col_order.append((model, f'AUC {split}', 'aucs'))
        col_order.append((model, f'FPR {split}', 'fprs'))

if __name__ == "__main__":
    dfs = {
            'aucs': pd.read_pickle(aucs_df_path),
            'fprs': pd.read_pickle(fprs_df_path),
            }

    # values[dataset][(analysis, reduction)][(model, column, metric)] = number
    values = {}
    for metric, df in dfs.items():
        for _, row in df.iterrows():
            entry = values.setdefault(row['dataset'], {}).setdefault((row['analysis'], row['reduction']), {})
            for _, col, _metric in col_order:
                if _metric == metric and col in row and pd.notna(row[col]):
                    entry[(row['model'], col, metric)] = row[col]

    template_lines = (results_path/'blank_auc_fpr_table.tex').read_text().splitlines()

    for dataset, ds_values in values.items():
        current_analysis = None
        new_lines = []
        for line in template_lines:
            if '\\caption{' in line:
                line = re.sub(r'\.\}', f' ({dataset}).}}', line)

            line = re.sub(r'\\label\{([^}]*)\}', lambda m: f'\\label{{{m.group(1)} {dataset}}}', line)

            am = re.search(r'\\(gls\{\w+\}|\w+Score)', line)
            if am and am.group(1) in analysis_macros:
                current_analysis = analysis_macros[am.group(1)]

            rm = re.search(r'\\(\w+DimRed)', line)
            reduction = reduction_macros[rm.group(1)] if rm else default_reduction

            if current_analysis is not None and '$.$' in line:
                entry = ds_values.get((current_analysis, reduction), {})

                originals = re.findall(r'\$([^$]*)\$', line)
                new_vals = list(originals)
                for i, (model, col, metric) in enumerate(col_order):
                    if (model, col, metric) in entry:
                        new_vals[i] = f'{entry[(model, col, metric)]:.2f}'

                it = iter(new_vals)
                line = re.sub(r'\$([^$]*)\$', lambda m: f'${next(it)}$', line)

            new_lines.append(line)

        (results_path/f'aucsFprs_{dataset}.tex').write_text('\n'.join(new_lines) + '\n')
        print(f'Table saved in {results_path/f"aucsFprs_{dataset}.tex"}')
