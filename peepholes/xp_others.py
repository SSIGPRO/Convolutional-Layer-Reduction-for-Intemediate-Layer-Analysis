# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

from statistics import geometric_mean as geomean
from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.plots.atks import auc_atks 

# --- softmax scores
from peepholelib.scores.model_confidence import MSPScore
from peepholelib.scores.doctor import DOCTORScore
from peepholelib.scores.relu import RelUScore

# --- logits scores 
from peepholelib.scores.energy import EnergyScore
from peepholelib.scores.max_logit import MaxLogitScore
from peepholelib.scores.predictive_entropy import PEScore

# --- input based scores
from peepholelib.featureSqueezing.FeatureSqueezingDetector import FeatureSqueezingDetector as FSD
from peepholelib.featureSqueezing.preprocessing import NLM_filtering_torch, bit_depth_torch, MedianPool2d
from peepholelib.scores.feature_squeezing import FeatureSqueezingScore

from configs.common import *
from utils.get_best_configs import test_configs 
from utils.save_aucs import save_aucs
    
if __name__ == "__main__":
    print(f'{args}') 
    lock_file = '../locks/peepholes.cuda.lock'
    lock = FileLock(lock_file)
    with lock.acquire(timeout=-1):
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        bs = int(bs_base*bs_model_scale*bs_red_scale*bs_analysis_scale)

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
    
    with datasets as ds, corevecs as cv, peepholes as ph:
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        score_loaders = [k for k in ds._dss.keys() if 'test' in k]
        ## SoftMax based scores
        MSPScore(save_path=scores_file)(
                datasets = ds,
                loaders = score_loaders,
                verbose = verbose,
                )

        DOCTORScore(save_path=scores_file)(
                datasets = ds,
                model = model,
                loaders = score_loaders,
                batch_size = int(bs_base*2),
                score_name = 'DOC',
                verbose = verbose,
                )
                                                              
        RelUScore(save_path=scores_file)(
                datasets = ds,
                loaders = score_loaders,
                fit_key = f'{args.dataset}-val-{args.model}',
                verbose = verbose,
                )

        ## Logits based scores 
        EnergyScore(save_path=scores_file)(
                datasets = ds,
                loaders = score_loaders,
                verbose = verbose,
                )

        MaxLogitScore(save_path=scores_file)(
                datasets = ds,
                loaders = score_loaders,
                verbose = verbose,
                )

        PEScore(save_path=scores_file)(
                datasets = ds,
                loaders = score_loaders,
                verbose = verbose,
                )

        ## Input based score

        fsd = FSD(
            model = model,
            prepro_dict = {
                'median': MedianPool2d(kernel_size=3, stride=1, padding=1),
                'bit_depth': partial(bit_depth_torch, bits=5),
                'nlm': partial(NLM_filtering_torch, kernel_size=11, std=4.0, kernel_size_mean=3, sub_filter_size=32),
                }
            )

        FeatureSqueezingScore(save_path=scores_file)(
                datasets = ds,
                loaders_ori = [k for k in score_loaders if 'test' in k],
                detector = fsd,
                batch_size = 2**6,
                score_name = 'FS',
                verbose = verbose,
                )
        
        # TODO: update aucs after PR
        auc_kwargs_ood = get_auc_kwargs_ood(args.model, args.dataset, ood_datasets)
        aucs_ood = auc_atks(
                datasets = ds,
                scores = scores,
                **auc_kwargs_ood,
                verbose = verbose
                )

        auc_kwargs_aa = get_auc_kwargs_aa(args.model, args.dataset, atk_names)
        aucs_aa = auc_atks(
                datasets = ds,
                scores = scores,
                **auc_kwargs_aa,
                verbose = verbose
                )

        report = {}
        _aucs = []
        for k in auc_kwargs_ood['atk_loaders']:
            report['AUC '+k] = list(aucs_ood[k].values())[0]
            _aucs.append(report['AUC '+k]) 
        report['AUC OoD'] = geomean(_aucs)
                                                             
        _aucs = []
        for k in auc_kwargs_aa['atk_loaders']:
            report['AUC '+k] = list(aucs_aa[k].values())[0]
            _aucs.append(report['AUC '+k]) 
        report['AUC AA'] = geomean(_aucs)
        
        # TODO: update after PR
        print('Report: ', report)
        save_aucs(
                report,
                aucs_df_path,
                dataset   = args.dataset,
                model     = args.model,
                reduction = '-',
                analysis  = args.analysis,
                )
