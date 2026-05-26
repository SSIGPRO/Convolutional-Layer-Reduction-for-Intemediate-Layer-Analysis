# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peephoelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors

from configs.common import *
    
if __name__ == "__main__":
    print(f'{args}') 
    lock_file = '../locks/corevectors.cuda.lock'
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
    # Dataset 
    #--------------------------------
    datasets = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # Reducers 
    #--------------------------------
    loaders = get_loaders(ood_datasets)
    inference_names = get_inference_names(ood_datasets)
    transforms = get_transforms(ood_datasets)

    with datasets as ds:
        ds.load_only(
                loaders = [f'{args.dataset}-train'],
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )
        sample_in = ds._dss[f'{args.dataset}-train-'+args.model][0]['image']

    reducers_kwargs = get_reducer_kwargs(model._target_modules) 
    reducers = {} 
    for _layer in target_layers:
        reducers[_layer] = Reducer(
                path = svds_path,
                model = model,
                layer = _layer,
                sample_in = sample_in,
                **reducers_kwargs[_layer],
                verbose = verbose
                ) 

    #--------------------------------
    # Corevectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            model = model,
            )
    
    with datasets as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        # computing the corevectors
        cv.get_coreVectors(
                datasets = ds,
                reducers = reducers,
                activations_parser = act_parser,
                save_input = save_input,
                save_output = save_output,
                names = cvs_names,
                batch_size = int(bs_base*bs_model_scale*bs_red_scale),
                n_threads = n_threads,
                verbose = verbose 
                )

        cv.normalize_corevectors(
                wrt = f'{args.dataset}-train-'+args.model,
                batch_size = int(bs_base*bs_red_scale),
                n_threads = n_threads,
                verbose=verbose
                )
