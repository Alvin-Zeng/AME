import json
import pathlib

from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_video

from .corrupt import ffmpeg_corrupt_dict


class SSV2(VisionDataset):
    def __init__(self, root:str, classes_path:str, annotation_path: str, debug=False, mode='ssv2'):
        """

        :param root:
        :param classes_path: path of a json which contains a dict {class:idx}
        :param annotation_path: path of a json which contains a array which i don't want to describe
        :param mode: 'ssv2','mini-ssv2'
        """
        super(SSV2, self).__init__(root)
        extension = ".mp4"
        self.classes = self._find_classes(classes_path, annotation_path, mode)  # dict
        self.video_paths, self.video_classes = self._get_index(annotation_path, extension, mode)
        self.length = len(self.video_paths) if not debug else 100

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item):
        video, audio, _ = self._get_video(item)
        class_index = self.video_classes[item]
        return video, audio, class_index

    @staticmethod
    def _find_classes(classes_path, annotation_path, mode):
        if mode == 'ssv2':
            with open(classes_path, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                data[k] = int(v)
            return data
        elif mode == 'mini-ssv2':
            with open(annotation_path, 'r') as f:
                data = json.load(f)['labels']  # list
            data = {k.replace('[', '').replace(']', '') : i for i, k in enumerate(data)}
            return data
        else:
            raise NotImplementedError

    def _get_index(self, path, extension, mode):
        with open(path, 'r') as f:
            data = json.load(f)
        if mode == 'ssv2':
            video_paths = [self.root + '/' + item['id'] + extension for item in data]
            video_classes = [self.classes[item["template"].replace('[', '').replace(']', '')] for item in data]
        elif mode == 'mini-ssv2':
            data = data['database']
            video_paths = list()
            video_classes = list()
            for k, v in data.items():
                video_paths.append(self.root + '/' + k + extension)
                video_classes.append(self.classes[v['annotations']['label'].replace('[', '').replace(']', '')])
        return video_paths, video_classes

    def _get_video(self, video_idx):
        from torchvision import get_video_backend
        backend = get_video_backend()

        video_path = self.video_paths[video_idx]

        if backend == "pyav":
            video, audio, info = read_video(video_path)
        else:
            raise NotImplementedError

        return video, audio, info


class SSV2C(VisionDataset):
    def __init__(self, root:str, corrupt_dir: str, corrupt_type: str, corrupt_severity: int,
                 classes_path: str, annotation_path: str, debug=False, mode='ssv2', is_corrupted=False):
        super(SSV2C, self).__init__(root)
        self.corrupt_dir = corrupt_dir
        self.corrupt_type = corrupt_type
        self.corrupt_severity = corrupt_severity
        self.classes_path = classes_path
        self.annotation_path = annotation_path
        self.debug = debug
        self.mode = mode
        self.is_corrupted = is_corrupted

        extension = ".mp4"
        self.classes = self._find_classes(classes_path, annotation_path, mode)  # dict
        self.video_paths, self.video_c_paths, self.video_classes = self._get_index(annotation_path, extension, mode)
        self.length = len(self.video_paths) if not debug else 100

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item):
        video, audio, _ = self._get_video(item)
        class_index = self.video_classes[item]
        return video, audio, class_index

    @staticmethod
    def _find_classes(classes_path, annotation_path, mode):
        if mode == 'ssv2':
            with open(classes_path, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                data[k] = int(v)
            return data
        elif mode == 'mini-ssv2':
            with open(annotation_path, 'r') as f:
                data = json.load(f)['labels']  # list
            data = {k.replace('[', '').replace(']', '') : i for i, k in enumerate(data)}
            return data
        else:
            raise NotImplementedError

    def _get_index(self, path, extension, mode):
        with open(path, 'r') as f:
            data = json.load(f)
        if mode == 'ssv2':
            corrupt_dir = pathlib.Path(self.corrupt_dir) / self.corrupt_type
            video_paths = [pathlib.Path(self.root + '/' + item['id'] + extension) for item in data]
            video_c_paths = [corrupt_dir / (item['id'] + extension) for item in data]
            video_classes = [self.classes[item["template"].replace('[', '').replace(']', '')] for item in data]
        elif mode == 'mini-ssv2':
            data = data['database']
            video_paths = list()
            video_c_paths = list()
            video_classes = list()
            for k, v in data.items():
                video_paths.append(pathlib.Path(self.root + '/' + k + extension))
                video_c_paths.append(pathlib.Path(self.corrupt_dir + '/' + k + extension))
                video_classes.append(self.classes[v['annotations']['label'].replace('[', '').replace(']', '')])
        return video_paths, video_c_paths, video_classes

    def _get_video(self, video_idx):
        from torchvision import get_video_backend
        backend = get_video_backend()

        video_path = self.video_paths[video_idx]
        video_c_path = self.video_c_paths[video_idx]
        if backend == "pyav":
            video, audio, info = read_video(str(video_c_path))
        else:
            raise NotImplementedError

        return video, audio, info





