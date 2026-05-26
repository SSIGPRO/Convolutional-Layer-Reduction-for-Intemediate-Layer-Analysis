# Python stuff
from functools import partial

# Torch stuff
from torch import linspace, int32

# Ray Stuff
from ray.tune import choice 

# Peepholelib stuff
from peepholelib.peepholes.classifiers.tgmm import GMM as Driller 
from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score 

bs_analysis_scale = 2**5

def get_drillers_kwargs(**kwargs):
    path = kwargs['path']
    name = kwargs['name']
    tl = kwargs['target_layers']
    nl_model = kwargs['nl_model']
    configs = kwargs['configs']
    device = kwargs['device']

    ret = {}
    for _l in tl:
        cv_dim = configs[_l]['cv_dim']
        n_clusters = configs[_l]['n_clusters']
        ret[_l] = {
                'path': path,
                'name': f'{name}.{_l}.{cv_dim}.{n_clusters}',
                'target_module': _l,
                'nl_classifier': n_clusters,
                'nl_model': nl_model,
                'n_features': cv_dim,
                'cls_kwargs': {
                    'covariance_regularization': 1e-4,
                    'convergence_tolerance': 1e-2
                    },
                'device': device
                } 
    return ret

def analysis_param_space(configs, args):
    for _n, _l in configs.items():
        if args.dataset == 'CIFAR100':
            _l['n_clusters'] = choice(linspace(50, 500, 10, dtype=int32).numpy().tolist())
        if args.dataset == 'ImageNet':
            _l['n_clusters'] = choice(linspace(50, 5000, 10, dtype=int32).numpy().tolist())
        
    configs['model'] = args.model
    configs['reduction'] = args.reduction
    configs['analysis'] = args.analysis
    configs['dataset'] = args.dataset
    return configs

# TODO: update score after PR
def get_score_fns(model, ds_name, ood_dss=None, atks=None, proto_threshold=0.9):
    return {
            'MACS': partial(
                proto_score,
                proto_key = f'{ds_name}-train-{model}',
                proto_threshold = proto_threshold
                )
        }

def get_auc_kwargs_ood(model, ds_name, ood_dss):
    return {
            'ori_loaders': {
                'MACS': f'{ds_name}-test-{model}',
                },
            'atk_loaders': [f'{k}-test-{model}' for k in ood_dss.keys()],
            'filter_key': None
            }

def get_auc_kwargs_aa(model, ds_name, atks):
    return {
            'ori_loaders': {
                'MACS': f'{ds_name}-test-{model}',
                },
            'atk_loaders': [
                f'{ds_name}-test-{a}-{model}' for a in atks],
            }
