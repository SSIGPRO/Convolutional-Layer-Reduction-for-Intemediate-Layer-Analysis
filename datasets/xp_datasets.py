import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

# python stuff
from functools import partial
from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peephoelib stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.samplers import balanced_subsampling as b_subs, random_subsampling as r_subs 
from peepholelib.datasets.functional.inference_fns import img_classification_full as img_cls_inf, img_classification_atks as img_cls_atk_inf 

# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.AutoAttack import myAutoAttack

from configs.common import *

lock_file = '../locks/datasets.cuda.lock'
if __name__ == "__main__":
    print(f'{args}') 
    lock = FileLock(lock_file)
    with lock.acquire(timeout=-1):
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        #------------------
        # Model 
        #------------------
        model = ModelWrap(
                model = Model(weights = pre_train_weights.DEFAULT),
                target_modules = target_layers,
                device = device
                )
    
    # in the non-imagenet case, we overwrite the model's
    # last layer and its weights
    if update_output:
        model.update_output(
                output_layer = output_layer, 
                to_n_classes = n_classes,
                overwrite = True 
                )
                                                
        model.load_checkpoint(
                path = model_path,
                name = model_name,
                verbose = True 
                )

    model.set_normalizer(
            mean = normalization_mean,
            std = normalization_std
            )

    #--------------------------------
    # Datasets 
    #--------------------------------
    # original datasets
    loaders = get_loaders(ood_datasets)
    transforms = get_transforms(ood_datasets)

    _dss = {
            args.dataset: Dataset(
                path = base_ds_path,
                seed = seed
                ),
           **ood_datasets
            }

    _dss_samplers = {}
    for _ds in _dss.keys():
        if _ds != 'Textures': # skip textures, 1880 samples only
            _dss_samplers[_ds] = partial(
                    b_subs if args.dataset in _ds else r_subs,
                    n_classes = n_classes,
                    n_samples = {
                        _ds+'-'+_s: _n for _s, _n in zip(['train', 'val', 'test'], [n_samples_train, n_samples_val, n_samples_test ]) if _ds+'-'+_s in loaders
                        }
                    )

    #######################
    # parsing datasets
    #######################
    
    # Instantiate DSs 
    dataset = ParsedDataset(
            path = ds_path,
            )

    # create inference functions for each atk
    # atks come from configs/common
    atks_inf_fns = {
            atk_name: partial(
                img_cls_atk_inf,
                attack = atk,
                label_key = 'label'
                ) for atk_name, atk in get_atks(model, robustbench_eps, attack_steps).items()
            }

    with dataset as ds:
        ds.parse_dataset(
                dataset_wraps = _dss,
                ds_samplers = _dss_samplers, 
                keys_to_copy = ['image', 'label'],
                batch_size = bs_base,
                n_threads = n_threads,
                chunk_size = chunk_size,
                verbose = verbose
                )

        ds.parse_inference(
                inference_fns = {args.model: partial(img_cls_inf, model=model)},
                transforms = transforms,
                batch_size = bs_base,
                n_threads = n_threads,
                verbose = verbose
                )

        # Apply attacks
        ds.parse_inference(
                loaders = [f'{args.dataset}-val', f'{args.dataset}-test'],
                inference_fns = atks_inf_fns, 
                transforms = transforms,
                batch_size = int(bs_base*bs_atk_scale),
                n_threads = n_threads,
                verbose = verbose 
                )
