import pandas as pd
from pathlib import Path


def saved_scores(path, dataset, model, reduction=None):
    '''
    Return the score names (`analysis` column) already saved in `path` for a
    given `dataset` and `model`, so that they are not recomputed. If
    `reduction` is passed, only the scores saved for that reduction are
    returned.
    '''
    path = Path(path)
    if not path.exists():
        return set()

    df = pd.read_pickle(path)
    mask = (df['dataset'] == dataset) & (df['model'] == model)
    if reduction != None:
        mask &= df['reduction'] == reduction

    return set(df.loc[mask, 'analysis'])


def save_metrics(report, path, cols, dataset, model, reduction, analysis):
    path = Path(path)

    row = {
            'dataset': dataset,
            'model': model,
            'reduction': reduction,
            'analysis': analysis,
            **{c: report[c] for c in cols},
            }

    if path.exists():
        df = pd.read_pickle(path)
        mask = (
                (df['dataset']   == dataset)  &
                (df['model']     == model)     &
                (df['reduction'] == reduction) &
                (df['analysis']  == analysis)
                )
        if mask.any():
            df.loc[mask, cols] = [row[c] for c in cols]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])

    df.to_pickle(path)


def save_aucs(report, path, dataset, model, reduction, analysis):
    save_metrics(report, path, ['AUC OoD', 'AUC AA', 'AUC general'], dataset, model, reduction, analysis)


def save_fprs(report, path, dataset, model, reduction, analysis):
    save_metrics(report, path, ['FPR OoD', 'FPR AA', 'FPR general'], dataset, model, reduction, analysis)
