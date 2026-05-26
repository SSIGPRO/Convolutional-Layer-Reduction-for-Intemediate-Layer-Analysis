from peepholelib.datasets.imagenet import ImageNet as Dataset
from peepholelib.datasets.iNaturalist import iNaturalist
from peepholelib.datasets.textures import Textures
from peepholelib.datasets.OpenImageO import OpenImageO

base_ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

inaturalist_path = '/srv/newpenny/dataset/iNaturalist'
textures_path = '/srv/newpenny/dataset/Textures'
openimage_path = '/srv/newpenny/dataset/OpenImageO'
ninco_path = '/srv/newpenny/dataset/NINCO'
ssbhard_path = '/srv/newpenny/dataset/SSB-Hard'

n_classes = 1000
update_output = False 
seed = 2

n_samples_train = 50000
n_samples_val = 5000
n_samples_test = 5000
robustbench_eps = 8/255
attack_steps = 500

proto_threshold = 0.0

ood_datasets = {
        'iNaturalist': iNaturalist(
            path = inaturalist_path,
            seed = seed
            ),
        'Textures': Textures(
            path = textures_path,
            seed = seed
            ),
        'OpenImageO': OpenImageO(
            path = openimage_path,
            seed = seed
            ),
        } 
