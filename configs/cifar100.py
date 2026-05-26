from peepholelib.datasets.cifar100 import Cifar100 as Dataset
from peepholelib.datasets.SVHN import SVHN
from peepholelib.datasets.Places import Places
from peepholelib.datasets.MNIST import MNIST
from peepholelib.datasets.textures import Textures

base_ds_path = '/srv/newpenny/dataset/CIFAR100'

svhn_path = '/srv/newpenny/dataset/SVHN'
places_path = '/srv/newpenny/dataset/Places365'
mnist_path = '/srv/newpenny/dataset/MNIST'
textures_path = '/srv/newpenny/dataset/DTD'

n_classes = 100
update_output = True
seed = 2

n_samples_train = 20000
n_samples_val = 5000
n_samples_test = 5000
robustbench_eps = 8/255
attack_steps = 500

proto_threshold = 0.9

ood_datasets = {
        'SVHN': SVHN(
            path = svhn_path,    
            seed = seed
            ),
        'Places365': Places(
            path = places_path,
            seed = seed
            ),
        'MNIST': MNIST(
            path = mnist_path,
            seed = seed
            ),
        'Textures': Textures(
            path = textures_path,
            seed = seed
            )
        }
