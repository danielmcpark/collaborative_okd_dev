import os
import tarfile
import scipy.io as io

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, check_integrity

__all__ = ['Dogs120Classification']

class Dogs120Classification(ImageFolder):
    img_folder = 'Images'
    img_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
    img_filename = 'images.tar'

    test_list = []
    num_training_classes = 120

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        
        if download:
            if not self._check_exists():
                download_url(self.img_url, self.root, self.img_filename)
                cwd = os.getcwd()
                tar = tarfile.open(os.path.join(self.root, self.train_img_filename), "r")
                os.chdir(self.root)
                tar.extractall()
                tar.close()
                os.chdir(cwd)
            else:
                print('Stanford dogs data were prepared..')

        ImageFolder.__init__(self, os.path.join(self.root),
                             transform=transform, target_transform=target_transform, **kwargs)
        self.root = root
        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]

        if train:
            labels = io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
            file_list = io.loadmat(os.path.join(self.root, 'train_list.mat'))['file_list']
        else:
            labels = io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']
            file_list = io.loadmat(os.path.join(self.root, 'test_list.mat'))['file_list']

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        class_idx = list(self.class_to_idx.values())
        samples = []
        assert len(labels) == len(file_list), 'Length is different'
        for l, p in zip(labels, file_list):
            cls = int(l[0]) - 1
            p = str(p[0][0])
            if cls in class_idx:
                samples.append((os.path.join(self.root, self.img_folder, p), int(cls)))

        self.samples = samples
        self.imgs = self.samples

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder))

if __name__=='__main__':
    dogs120 = Dogs120Classification('/mnt/disk3/dogs120/', train=True, download=True)
    #cars196 = Cars196Metric('/mnt/disk3/cars196/', train=True, download=False)
    print(len(dogs120))
