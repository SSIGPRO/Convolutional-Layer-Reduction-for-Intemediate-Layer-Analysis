# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

import pandas as pd
import torch

def get_best_config(hyperp_file):
    hf = Path(hyperp_file)
    _df = pd.read_pickle(hf)
    print(hyperp_file, len(_df))
   
    best = _df[['AUC general']].values.argmax()
    best_config = _df.iloc[best:best+1]

    return best_config

if __name__ == '__main__':
    bc = get_best_config()

def test_configs(tl, file):
    ret = {}
    bc = get_best_config(file)
    # tunned convolution configs
    for _n, _l in tl.items():
        ret[_n] = {}
        for _c in ['cv_dim', 'n_clusters', 'magnitude']:
            if _n+'/'+_c in bc:
                ret[_n][_c] = bc[_n+'/'+_c].iloc[0]

        # avg_pooling with MACS case
        if _n+'/cv_dim' not in bc:
            ret[_n]['cv_dim'] = _l.out_channels 

    return ret
