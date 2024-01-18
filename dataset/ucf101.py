import os
import pathlib
import time

import tqdm
import torchvision
import torch
from torch.utils.data import Dataset

import os
from typing import Optional, Callable, Tuple, Dict, Any, List
from torch import Tensor
import logging
import PIL

from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.vision import VisionDataset
from torchvision import get_video_backend
from torchvision.io import read_video


from typing import Any, Dict, List, Tuple, Optional, Callable
from torch import Tensor

from .video import VideoClips

class UCF101(Dataset):
    def __init__(self, root: str, classes_path: str, annotation_path: str,
                 frame_rate: Optional[int] = None, extension = '.avi', debug = False):
        super(UCF101, self).__init__()
        self.root = pathlib.Path(root)
        self.extension = extension
        self.classes_path = pathlib.Path(classes_path)
        self.annotation_path = pathlib.Path(annotation_path)
        self.classes = self._find_classes()  # dict
        self.video_paths, self.video_classes = self._get_index()
        self.debug = debug
        self.length = len(self.video_paths) if not debug else min(100, len(self.video_paths))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        video, audio, _ = self._get_video(item)
        class_index = self.video_classes[item]
        return video, audio, class_index

    def _find_classes(self):
        with open(self.classes_path, 'r') as f:
            data = f.readlines()
        data = {x.strip().split()[1]:i for i, x in enumerate(data)}
        return data

    def _get_index(self):
        if self.annotation_path.is_dir():
            annotation_file_path = self.annotation_path / 'testlist01.txt'
        else:
            annotation_file_path = self.annotation_path

        with open(annotation_file_path, 'r') as f:
            data = f.readlines()
        data = [pathlib.Path(x.strip().split()[0]) for x in data]
        video_paths = list()
        video_classes = list()
        for p in data:
            p = p.with_suffix(self.extension)
            video_paths.append(self.root / p)
            video_classes.append(self.classes[str(p.parent)])
        assert len(video_paths) == len(video_classes)
        return video_paths, video_classes

    def _get_video(self, video_idx):

        backend = get_video_backend()

        video_path = self.video_paths[video_idx]

        if backend == "pyav":
            video, audio, info = read_video(str(video_path))
            assert video.shape[0] > 0, f"path:{str(video_path)}"
        else:
            raise NotImplementedError

        return video, audio, info












