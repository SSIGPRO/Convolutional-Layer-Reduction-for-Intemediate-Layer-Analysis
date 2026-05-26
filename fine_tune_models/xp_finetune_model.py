import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

from datetime import datetime

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peepholelibe stuff
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.training.trainingBase import Trainer

# Our stuff
from configs.common import *
from configs.fine_tune_models import *

if __name__ == "__main__":
    print(f'{args}') 
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_dir = args.data_path/"checkpoints"/args.model 
    tune_name = 'model'

    #--------------------------------
    # Dataset 
    #--------------------------------

    dataset = Cifar100(
            path = base_ds_path,
            std_transform = transform,
            aug_transform = augmentation
            )

    dataset.__load_data__()

    #--------------------------------
    # Model 
    #--------------------------------

    nn = Model(weights=pre_train_weights.DEFAULT)

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = output_layer,
            to_n_classes = n_classes
            )

    model.set_normalizer(
            mean = normalization_mean,
            std = normalization_std
            )

    #-----------------------------------------------
    # Finetune the model
    #-----------------------------------------------
    
    # after model is created / normalized
    trainable_params = model.get_trainable_parameters(
            layers_to_train = None,
            verbose = verbose
            )

    optimizer = Optim(
            trainable_params,
            **opt_kwargs,
            )
    
    scheduler = Scheduler(
            optimizer,
            **scheduler_kwargs
            )

    bs = int(model_fine_tune_bs*bs_model_scale)

    datasets = {
            'train': dataset.__dataset__[f'{dataset_name}-train'],
            'val': dataset.__dataset__[f'{dataset_name}-val'],
            'test': dataset.__dataset__[f'{dataset_name}-test'],
            }

    dataloader_kwargs = {
            'train': dict(batch_size = bs, shuffle = True, **dl_kwargs),
            'val': dict(batch_size = bs, shuffle = False, **dl_kwargs),
            'test': dict(batch_size = bs, shuffle = False, **dl_kwargs),
            }

    iterations_kwargs = {'train': iterations, 'val': iterations, 'test': 'full'}

    finetuner = Trainer(
            model = model,
            path = tune_dir,
            name = tune_name,
            datasets = datasets,
            dataloader_kwargs = dataloader_kwargs,
            iterations = iterations_kwargs,
            max_epochs = max_epochs,
            optimizer = optimizer,
            scheduler = scheduler,
            early_stopping_patience = max_epochs,
            save_every = save_every,
            verbose = verbose
            )
    
    finetuner.fit()
    finetuner.test()
