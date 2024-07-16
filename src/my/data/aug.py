from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageFilter
from .simclr_transform import GaussianBlur


class AugPairDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        x1, x2 = self.transform(x), self.transform(x)
        return x1, x2

    def __len__(self) -> int:
        return len(self.dataset)



class AugPairRotDataset(Dataset):
    def __init__(self, dataset, transform):
        super(AugPairRotDataset).__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        n = random.random()
        angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
        # angle = 0 if n <= 0.125 else 1 if n <= 0.25 else 2 if n <= 0.375 else 3 if n <= 0.5 else 4 if n <= 0.625 else 5 if n <= 0.75 else 6 if n <= 0.875 else 7

        x1, x2 = self.transform(x), self.transform(x)
        x3 = image_rot(self.transform(x), 90 * angle)
        return x1, x2, x3, angle

    def __len__(self) -> int:
        return len(self.dataset)

class AugTripleDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        x1, x2, x3 = self.transform(x), self.transform(x), self.transform(x)
        return x, x1, x2, x3

    def __len__(self) -> int:
        return len(self.dataset)


def image_rot(image, angle):
    image = TF.rotate(image, angle)
    return image


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class AugOrchestraDataset(Dataset):
    def __init__(self, dataset, is_sup):
        super(AugOrchestraDataset, self).__init__()
        image_size = 32
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_prime = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            Solarization(p=0.2),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = dataset

        self.mode = is_sup

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        n = random.random()
        angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
        # angle = 0 if n <= 0.125 else 1 if n <= 0.25 else 2 if n <= 0.375 else 3 if n <= 0.5 else 4 if n <= 0.625 else 5 if n <= 0.75 else 6 if n <= 0.875 else 7
        if (self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform_prime(x)
            x3 = image_rot(self.transform(x), 90 * angle)
            # x3 = image_rot(self.transform(x), 45 * angle)
            return x1, [x2, x3, angle]

    def __len__(self) -> int:
        return len(self.dataset)


class AugSimclrDataset(Dataset):
    def __init__(self, dataset, is_sup):
        super(AugSimclrDataset, self).__init__()
        image_size = 32
        s = 1  # n_views??
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transform = T.Compose([T.RandomResizedCrop(size=image_size),
                                    T.RandomHorizontalFlip(),
                                    T.RandomApply([color_jitter], p=0.8),
                                    T.RandomGrayscale(p=0.2),
                                    GaussianBlur(kernel_size=int(0.1 * image_size)),
                                    T.ToTensor()])

        self.dataset = dataset

        self.mode = is_sup

    def __getitem__(self, index: int):
        x, _ = self.dataset[index]
        n = random.random()
        # angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
        angle = 0 if n <= 0.125 else 1 if n <= 0.25 else 2 if n <= 0.375 else 3 if n <= 0.5 else 4 if n <= 0.625 else 5 if n <= 0.75 else 6 if n <= 0.875 else 7
        if (self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform(x)
            # x3 = image_rot(self.transform(x), 90 * angle)
            x3 = image_rot(self.transform(x), 45 * angle)
            return x1, [x2, x3, angle]

    def __len__(self) -> int:
        return len(self.dataset)
