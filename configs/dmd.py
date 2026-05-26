# Python stuff
from functools import partial

# Ray Stuff
from torch import linspace
from ray.tune import choice 

# Peepholelib stuff
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as Driller 
from peepholelib.scores.dmd import DMD_score as dmd_score

bs_analysis_scale = 2**-2

def get_drillers_kwargs(**kwargs):
    path = kwargs['path']
    name = kwargs['name']
    tl = kwargs['target_layers']
    nl_model = kwargs['nl_model']
    model = kwargs['model']
    configs = kwargs['configs']
    act_parser = kwargs['act_parser']
    save_input = kwargs['save_input']
    save_output = kwargs['save_output']
    device = kwargs['device']

    ret = {}
    for _l in tl:
        cv_dim = configs[_l]['cv_dim']
        mag = configs[_l]['magnitude']
        ret[_l] = {
                'path': path,
                'name': f'{name}.{_l}.{cv_dim}.{mag}', 
                'target_module': _l,
                'nl_model': nl_model,
                'n_features': cv_dim,
                'model': model,
                # divide by 1000 to avoid large file names
                'magnitude': mag/1000,
                'std_transform': [0.229, 0.224, 0.225],
                'act_parser': act_parser,
                'save_input': save_input,
                'save_output': save_output,
                'device': device
                } 
    return ret

def analysis_param_space(configs, args):
    for _n, _l in configs.items():
        # mag is divided by 1000 at dmd to avoid large file names
        _l['magnitude'] = choice(linspace(1, 10, 10).numpy().tolist())
    configs['model'] = args.model
    configs['reduction'] = args.reduction
    configs['analysis'] = args.analysis
    configs['dataset'] = args.dataset
    return configs

# TODO: update score after PR
def get_score_fns(model, ds_name, ood_dss, atks, **kwargs):
    return {
            'DMD-ood': partial(
                dmd_score,
                pos_loader_train = f'{ds_name}-val-{model}',
                pos_loader_test = f'{ds_name}-test-{model}',
                neg_loaders = {f'{k}-test-{model}': [f'{k}-val-{model}'] for k in ood_dss.keys()},
                ),
            'DMD-aa': partial(
                dmd_score,
                pos_loader_train = f'{ds_name}-val-{model}',
                pos_loader_test = f'{ds_name}-test-{model}',
                neg_loaders = {f'{ds_name}-test-{a}-{model}': [f'{ds_name}-val-{a}-{model}'] for a in atks},
                ),
        }

def get_auc_kwargs_ood(model, ds_name, ood_dss):
    return {
            'ori_loaders': {
                'DMD-ood': [f'{k}-val-{model}' for k in ood_dss.keys()],
                },
            'atk_loaders': [f'{k}-test-{model}' for k in ood_dss.keys()],
            'filter_key': None
            }

def get_auc_kwargs_aa(model, ds_name, atks):
    return {
            'ori_loaders': {
                'DMD-aa': [f'{ds_name}-val-{a}-{model}' for a in atks],
                },
            'atk_loaders': [f'{ds_name}-test-{a}-{model}' for a in atks],
            }
