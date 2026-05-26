import pandas as pd
from pathlib import Path


def save_aucs(report, path, dataset, model, reduction, analysis):
    path = Path(path)
    row = {
            'dataset': dataset,
            'model': model,
            'reduction': reduction,
            'analysis': analysis,
            'AUC OoD': report['AUC OoD'],
            'AUC AA': report['AUC AA'],
            }

    key_cols = ['dataset', 'model', 'reduction', 'analysis']

    if path.exists():
        df = pd.read_pickle(path)
        mask = (
                (df['dataset']   == dataset)  &
                (df['model']     == model)     &
                (df['reduction'] == reduction) &
                (df['analysis']  == analysis)
                )
        if mask.any():
            df.loc[mask, ['AUC OoD', 'AUC AA']] = row['AUC OoD'], row['AUC AA']
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])

    df.to_pickle(path)
