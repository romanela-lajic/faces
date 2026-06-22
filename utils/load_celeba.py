
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
import sklearn.datasets as sklearn_datasets
from timm.data import create_transform
from torch.utils.data import TensorDataset, Subset

from PIL import Image
import pandas as pd

import pdb


import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple

import PIL
import torch

from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg, extract_archive
from torchvision.datasets.vision import VisionDataset

CSV = namedtuple("CSV", ["header", "index", "data"])


class FullCelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")



        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target


    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)





def interpolation_flag(interpolation):
    if interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    raise ValueError("interpolation must be one of 'bilinear', 'bicubic'")




def full_celeba_get_datasets(data_dir, use_data_aug=True, label_indices=None,   interpolation='bilinear', return_test=False, **kwargs):
    interpolation = interpolation_flag(interpolation)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If we are only using some of the labels, remove all the ones we don't need.
    target_transform=None
    if label_indices:
        target_transform = lambda x: x[[label_indices]]

    if use_data_aug:
        # See https://github.com/princetonvisualai/DomainBiasMitigation/blob/c432e751632bce2c7467ef22a6ffb44402b88684/models/celeba_core.py#L53
        train_transform = transforms.Compose([
                                              transforms.Resize(112),
                                              transforms.RandomCrop(112),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                                              transforms.RandomGrayscale(p=0.05),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(112, interpolation=interpolation),
                                              transforms.CenterCrop(112),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
        
    transform=create_transform(
            input_size=112,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
           mean=[0.5, 0.5, 0.5],
           std=[0.5, 0.5, 0.5])
    # assumes the full CelebA data is already downloaded and unzipped -- otherwise unzipping is not implemented, since the format is 7z, not python friendly
    train_dataset = FullCelebA(root=data_dir, split='train', target_type='attr', transform=transform, target_transform = target_transform)

    #train_dataset = datasets.CelebA(root=data_dir, split='train', target_type='attr', transform=train_transform, target_transform=target_transform, download=True)

    test_transform = transforms.Compose([
        transforms.Resize(112, interpolation=interpolation),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = FullCelebA(root=data_dir, split='valid', target_type='attr', transform=test_transform, target_transform = target_transform)
#datasets.CelebA(root=data_dir, split='valid', target_type='attr', transform=test_transform, target_transform=target_transform, download=True)
    if not return_test:
        return train_dataset, val_dataset
    test_dataset = FullCelebA(root=data_dir, split='test', target_type='attr', transform=test_transform, target_transform = target_transform)
#datasets.CelebA(root=data_dir, split='test', target_type='attr', transform=test_transform, target_transform=target_transform, download=True)
    return train_dataset, val_dataset, test_dataset


import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import random

def generate_celeba_pairs(file_path, output_path, num_negatives=1):
    """
    Generates positive and negative pairs from the CelebA dataset and saves them to a file.

    Args:
        file_path (str): Path to the identity list file.
        output_path (str): Path where the pairs will be saved.
        num_negatives (int): Number of negative samples per positive pair.
    """
    labels = {}

    # Read the file and organize data
    with open(file_path, 'r') as f:
        for line in f:
            image, identity = line.strip().split()
            if identity not in labels:
                labels[identity] = []
            labels[identity].append(image)

    # Generate positive pairs
    pairs = []
    for identity, images in labels.items():
        if len(images) > 1:
            pairs.extend([(img1, img2, 1) for img1, img2 in combinations(images, 2)])

    # Generate negative pairs
    all_identities = list(labels.keys())
    for _ in range(num_negatives * len(pairs)):
        id1, id2 = random.sample(all_identities, 2)
        img1 = random.choice(labels[id1])
        img2 = random.choice(labels[id2])
        pairs.append((img1, img2, 0))

    # Write to file
    with open(output_path, 'w') as f:
        for img1, img2, label in pairs:
            f.write(f"{img1} {img2} {label}")



# def get_dataloader(file_path, batch_size=32, num_workers=4):
#     """
#     Returns a DataLoader for the CelebA pairs dataset.

#     Args:
#         file_path (str): Path to the identity list file.
#         batch_size (int): Number of samples per batch.
#         num_workers (int): Number of subprocesses for data loading.

#     Returns:
#         DataLoader: Iterable DataLoader for model training or evaluation.
#     """
#     dataset = CelebAPairsDataset(file_path)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)



