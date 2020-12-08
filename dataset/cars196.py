import os
import tarfile
import scipy.io as io

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, check_integrity

__all__ = ['Cars196Metric', 'Cars196Classification']

class Cars196Metric(ImageFolder):
    base_folder = 'car_ims'
    img_url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
    img_filename = 'car_ims.tgz'
    img_md5 = 'd5c8f0aa497503f355e17dc7886c3f14'

    anno_url = 'http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
    anno_filename = 'cars_annos.mat'
    anno_md5 = 'b407c6086d669747186bd1d764ff9dbc'

    checklist = [
        ['016185.jpg', 'bab296d5e4b2290d024920bf4dc23d07'],
        ['000001.jpg', '2d44a28f071aeaac9c0802fddcde452e'],
    ]

    test_list = []
    num_training_classes = 98

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        if download:
            download_url(self.img_url, self.root, self.img_filename, self.img_md5)
            download_url(self.anno_url, self.root, self.anno_filename, self.anno_md5)

            if not self._check_integrity():
                cwd = os.getcwd()
                tar = tarfile.open(os.path.join(self.root, self.img_filename), "r:gz")
                os.chdir(self.root)
                tar.extractall()
                tar.close()
                os.chdir(cwd)

        if not self._check_integrity() or \
           not check_integrity(os.path.join(self.root, self.anno_filename), self.anno_md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        ImageFolder.__init__(self, os.path.join(self.root),
                             transform=transform, target_transform=target_transform, **kwargs)
        self.root = root

        labels = io.loadmat(os.path.join(self.root, self.anno_filename))['annotations'][0]
        class_names = io.loadmat(os.path.join(self.root, self.anno_filename))['class_names'][0]

        if train:
            self.classes = [str(c[0]) for c in class_names[:98]]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.classes = [str(c[0]) for c in class_names[98:]]
            self.class_to_idx = {cls: i+98 for i, cls in enumerate(self.classes)}

        class_idx = list(self.class_to_idx.values())
        samples = []
        for l in labels:
            cls = int(l[5][0, 0]) - 1
            p = l[0][0]
            print(l[0][0])
            if cls in class_idx:
                samples.append((os.path.join(self.root, p), int(cls)))

        self.samples = samples
        self.imgs = self.samples

    def _check_integrity(self):
        for f, md5 in self.checklist:
            fpath = os.path.join(self.root, self.base_folder, f)
            if not check_integrity(fpath, md5):
                return False
        return True

class Cars196Classification(ImageFolder):
    train_folder = 'cars_train'
    train_img_url = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    train_img_filename = 'car_train.tgz'
    train_img_md5 = '065e5b463ae28d29e77c1b4b166cfe61'

    test_folder = 'cars_test'
    test_img_url = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    test_img_filename = 'cars_test.tgz'
    test_img_md5 = '4ce7ebf6a94d07f1952d94dd34c4d501'

    test_list = []
    num_training_classes = 196

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        if download:
            if not self._check_exists():
                if train:
                    download_url(self.train_img_url, self.root, self.train_img_filename, self.train_img_md5)
                    cwd = os.getcwd()
                    tar = tarfile.open(os.path.join(self.root, self.train_img_filename), "r:gz")
                else:
                    download_url(self.test_img_url, self.root, self.test_img_filename, self.test_img_md5)
                    cwd = os.getcwd()
                    tar = tarfile.open(os.path.join(self.root, self.test_img_filename), "r:gz")
                os.chdir(self.root)
                tar.extractall()
                tar.close()
                os.chdir(cwd)
            else:
                print('Stanford cars data prepared..')

        ImageFolder.__init__(self, os.path.join(self.root),
                             transform=transform, target_transform=target_transform, **kwargs)
        self.root = root

        if train:
            labels = io.loadmat(os.path.join(self.root, 'devkit/cars_train_annos.mat'))['annotations'][0]
        else:
            labels = io.loadmat(os.path.join(self.root, 'devkit/cars_test_annos_withlabels.mat'))['annotations'][0]
        class_names = io.loadmat(os.path.join(self.root, 'devkit/cars_meta.mat'))['class_names'][0]

        self.classes = [str(c[0]) for c in class_names]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        class_idx = list(self.class_to_idx.values())
        samples = []
        for l in labels:
            cls = int(l[4][0, 0]) - 1
            p = l[5][0]
            if cls in class_idx:
                samples.append((os.path.join(self.root, self.train_folder, p), int(cls))) if train else \
                        samples.append((os.path.join(self.root, self.test_folder, p), int(cls)))

        self.samples = samples
        self.imgs = self.samples

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.train_folder)) and os.path.exists(os.path.join(self.root, self.test_folder)))

if __name__=='__main__':
    cars196 = Cars196Classification('/mnt/disk3/cars196/', train=False, download=False)
    #cars196 = Cars196Metric('/mnt/disk3/cars196/', train=True, download=False)
    print(len(cars196))
