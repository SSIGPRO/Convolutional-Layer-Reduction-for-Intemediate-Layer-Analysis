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
from peepholelib.plots.atks import auc_fpr

# --- softmax scores
from peepholelib.scores.model_confidence import model_confidence_score
from peepholelib.scores.doctor import DOCTOR_score
from peepholelib.scores.relu import RelU_score

# --- input based scores
from peepholelib.featureSqueezing.FeatureSqueezingDetector import FeatureSqueezingDetector as FSD
from peepholelib.featureSqueezing.preprocessing import NLM_filtering_torch, bit_depth_torch, MedianPool2d
from peepholelib.scores.feature_squeezing import feature_squeezing_score

from configs.common import *
from utils.get_best_configs import test_configs
from utils.save_aucs import save_aucs, save_fprs, saved_scores

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

    loaders = get_loaders(ood_datasets)
    inference_names = get_inference_names(ood_datasets)
    transforms = get_transforms(ood_datasets)

    #--------------------------------
    # Dataset
    #--------------------------------
    datasets = ParsedDataset(
            path = ds_path,
            )

    with datasets as ds:
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        score_loaders = [k for k in ds._dss.keys() if 'test' in k]

        _saved = saved_scores(aucs_df_path, args.dataset, args.model)
        print('Scores already saved: ', sorted(_saved))

        scores = {}

        ## SoftMax based scores
        if not 'MSP' in _saved:
            scores = model_confidence_score(
                    datasets = ds,
                    loaders = score_loaders,
                    score_name = 'MSP',
                    append_scores = scores,
                    verbose = verbose,
                    )

        if not 'DOC' in _saved:
            scores = DOCTOR_score(
                    datasets = ds,
                    model = model,
                    loaders = score_loaders,
                    batch_size = int(bs_base*2),
                    score_name = 'DOC',
                    append_scores = scores,
                    verbose = verbose,
                    )

        if not 'Rel-U' in _saved:
            scores = RelU_score(
                    datasets = ds,
                    loaders = score_loaders,
                    fit_key = f'{args.dataset}-val-{args.model}',
                    score_name = 'Rel-U',
                    append_scores = scores,
                    verbose = verbose,
                    )

        ## Input based score
        if not 'FS' in _saved:
            fsd = FSD(
                model = model,
                prepro_dict = {
                    'median': MedianPool2d(kernel_size=3, stride=1, padding=1),
                    'bit_depth': partial(bit_depth_torch, bits=5),
                    'nlm': partial(NLM_filtering_torch, kernel_size=11, std=4.0, kernel_size_mean=3, sub_filter_size=32),
                    }
                )

            scores = feature_squeezing_score(
                    datasets = ds,
                    loaders_ori = score_loaders,
                    detector = fsd,
                    batch_size = 2**6,
                    score_name = 'FS',
                    append_scores = scores,
                    verbose = verbose,
                    )

        #--------------------------------
        # AUCs
        #--------------------------------
        _ori_loader = f'{args.dataset}-test-{args.model}'

        score_names = list(scores.get(_ori_loader, {}).keys())
        print('Computing AUCs for: ', score_names)

        auc_kwargs_ood = get_auc_kwargs_ood(args.model, args.dataset, ood_datasets)
        auc_kwargs_ood['ori_loaders'] = {_sn: _ori_loader for _sn in score_names}
        aucs_ood, fprs_ood = auc_fpr(
                datasets = ds,
                scores = scores,
                **auc_kwargs_ood,
                verbose = verbose
                )

        auc_kwargs_aa = get_auc_kwargs_aa(args.model, args.dataset, atk_names)
        auc_kwargs_aa['ori_loaders'] = {_sn: _ori_loader for _sn in score_names}
        aucs_aa, fprs_aa = auc_fpr(
                datasets = ds,
                scores = scores,
                **auc_kwargs_aa,
                verbose = verbose
                )

        for score_name in score_names:
            report = {}
            _aucs_ood = []
            for k in auc_kwargs_ood['atk_loaders']:
                report['AUC '+k] = aucs_ood[k][score_name]
                _aucs_ood.append(report['AUC '+k])
            report['AUC OoD'] = geomean(_aucs_ood)

            _aucs_aa = []
            for k in auc_kwargs_aa['atk_loaders']:
                report['AUC '+k] = aucs_aa[k][score_name]
                _aucs_aa.append(report['AUC '+k])
            report['AUC AA'] = geomean(_aucs_aa)

            report['AUC general'] = geomean(_aucs_ood+_aucs_aa)

            print(f'Report {score_name}: ', report)
            save_aucs(
                    report,
                    aucs_df_path,
                    dataset   = args.dataset,
                    model     = args.model,
                    reduction = '-',
                    analysis  = score_name,
                    )

            # `auc_fpr()` returns 1-FPR, which is aggregated with the geometric mean
            fpr_report = {}
            _fprs_ood = []
            for k in auc_kwargs_ood['atk_loaders']:
                _fprs_ood.append(fprs_ood[k][score_name])
                fpr_report['FPR '+k] = 1 - _fprs_ood[-1]
            fpr_report['FPR OoD'] = 1 - geomean(_fprs_ood)

            _fprs_aa = []
            for k in auc_kwargs_aa['atk_loaders']:
                _fprs_aa.append(fprs_aa[k][score_name])
                fpr_report['FPR '+k] = 1 - _fprs_aa[-1]
            fpr_report['FPR AA'] = 1 - geomean(_fprs_aa)

            fpr_report['FPR general'] = 1 - geomean(_fprs_ood+_fprs_aa)

            print(f'FPR report {score_name}: ', fpr_report)
            save_fprs(
                    fpr_report,
                    fprs_df_path,
                    dataset   = args.dataset,
                    model     = args.model,
                    reduction = '-',
                    analysis  = score_name,
                    )
