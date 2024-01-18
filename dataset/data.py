from pathlib import Path
import numpy as np
import torch
import torchvision
import random
import math
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video
from numpy.random import randint

from .corrupt import Corrupt, ffmpeg_corrupt_dict


class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        data = [tsfm(x) for tsfm in self.transform_list]
        return data


class GroupScale(object):
    def __init__(self, size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.RandomCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)  # [H,W,C*batch_size]


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return tensor


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]  # [224, 224]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):  # input list of PIL Images

        im_size = img_group[0].size  # W H C ?

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]  # 441 331

        # find a crop size
        base_size = min(image_w, image_h)  # 331
        crop_sizes = [int(base_size * x) for x in self.scales]  # 331 * [1, .875, .75, .66] = [331, 289, 248, 218]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        # [240, 202, 172, 158] [331, 289, 248, 218]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
        # [240, 202, 172, 158] [331, 289, 248, 218]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:  # <= 1
                    pairs.append((w, h))
                    # (331,331)(331,289)(289,331)(289,289)(289,248)(248,289)(248,248)(248,218)(218,248)(218,218)

        crop_pair = random.choice(pairs)  # ?
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])  # 441 331

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class Mask(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, video):
        video = np.array(video)
        if len(video.shape)==5:
            batch_mask_video = []
            batch = len(video)
            num_clip = len(video[0])
            for i in range(batch):
                mask_video = []
                for j in range(num_clip):
                    clip = np.array(video[i][j])
                    lam = torch.full(clip.shape, self.alpha)
                    clip = lam * clip 
                    mask_video.append(Image.fromarray(np.uint8(clip)))
                batch_mask_video.append(mask_video)
            return batch_mask_video
        else:
            mask_video = []
            num_clip = len(video)
            for i in range(num_clip):
                clip = np.array(video[i])
                lam = torch.full(clip.shape, self.alpha)
                clip = lam * clip
                mask_video.append(Image.fromarray(np.uint8(clip)))
            return mask_video


class Trans(object):
    def __init__(self, corrupt_type, corrupt_severity, corrupt_first=True ,**kwargs):
        scale_size = kwargs.get('scale_size', 256)
        input_size = kwargs.get('input_size', 224)
        roll = kwargs.get('roll', False)
        div = kwargs.get('div', True)
        input_mean = kwargs.get('input_mean', [0.485, 0.456, 0.406])
        input_std = kwargs.get('input_std', [0.229, 0.224, 0.225])
        self.transform = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
            Corrupt(corrupt_type, corrupt_severity),
            Stack(roll),
            ToTorchFormatTensor(div),
            GroupNormalize(input_mean, input_std)
        ])

    def __call__(self, clips):
        if len(clips.shape) == 4:
            clips = clips.reshape(1, *clips.shape)
        return torch.stack([self.transform([Image.fromarray(np.uint8(img)) for img in clip]) for clip in clips])  # Tensor[2,N*C,S,S]


class AdaptTrans(object):
    def __init__(self, corrupt_type, corrupt_severity, corrupt_first, **kwargs):
        scale_size = kwargs.get('scale_size', 256)
        input_size = kwargs.get('input_size', 224)
        roll = kwargs.get('roll', False)
        div = kwargs.get('div', True)
        input_mean = kwargs.get('input_mean', [0.485, 0.456, 0.406])
        input_std = kwargs.get('input_std', [0.229, 0.224, 0.225])
        self.transform = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
            Corrupt(corrupt_type, corrupt_severity),
            Stack(roll),
            ToTorchFormatTensor(div),
            GroupNormalize(input_mean, input_std)
        ])
        self.scale_size = scale_size
        self.input_size = input_size
        self.corrupt_type = corrupt_type
        self.corrupt_severity = corrupt_severity
        self.roll = roll
        self.div = div
        self.input_mean = input_mean
        self.input_std = input_std
        self.augmentation = self.get_augmentation_versions()

    def __call__(self, clips):
        if len(clips.shape) == 4:
            clips = clips.reshape(1, *clips.shape)
        data =[]
        for aug in self.augmentation:
            aug_data = torch.stack([aug([Image.fromarray(np.uint8(img)) for img in clip]) for clip in clips])
            data.append(aug_data)
        return torch.stack(data)
    
    def get_augmentation_versions(self):
        """
        Get a list of augmentations. "w" stands for weak, "s" stands for strong.

        E.g., "wss" stands for one weak, two strong.
        """
        transform_list = []
        transform_list.append(self.get_augmentation("s"))
        transform_list.append(self.get_augmentation("w"))
        transform_list.append(self.get_augmentation("w"))
        # transform = NCropsTransform(transform_list)
        # return transform
        return transform_list
    
    def get_augmentation(self, aug_type, normalize=None):
        if not normalize:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        if aug_type == "w":
            return transforms.Compose(
                [
                    GroupScale(self.scale_size),
                    GroupCenterCrop(self.input_size),
                    Corrupt(self.corrupt_type, self.corrupt_severity),
                    Stack(self.roll),
                    ToTorchFormatTensor(self.div),
                    GroupNormalize(self.input_mean, self.input_std)
                ]
            )
        elif aug_type == "s":
            return transforms.Compose(
                [
                    GroupScale(self.scale_size),
                    GroupCenterCrop(self.input_size),
                    Corrupt(self.corrupt_type, self.corrupt_severity),
                    Mask(0.9),
                    Stack(self.roll),
                    ToTorchFormatTensor(self.div),
                    GroupNormalize(self.input_mean, self.input_std)
                ]
            )
        return None


class SlowFastDataset(Dataset):
    def __init__(self, base_dataset, corrupt_type, corrupt_severity, num_segments, **kwargs):
        self.base_dataset = base_dataset
        self.num_segments = num_segments
        self.kwargs = kwargs
        is_corrupted = kwargs.pop('is_corrupted', False)
        self.corrupt_type = 'origin' if is_corrupted else corrupt_type
        self.corrupt_severity = corrupt_severity
        corrupt_first = kwargs.pop('corrupt_first', False)
        self.adapt_transform = AdaptTrans(self.corrupt_type, self.corrupt_severity, corrupt_first, **kwargs)
        self.test_transform = Trans(self.corrupt_type, self.corrupt_severity, corrupt_first, **kwargs)
        self.train_mode = False

    def __len__(self):
        return self.base_dataset.__len__()

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def __getitem__(self, item):
        if self.train_mode:
            video, audio, label = self.base_dataset.__getitem__(item)  # video should be Tensor or ndarray
            slow_clips, fast_clips = self.__temporal_augment(video)  # ndarray[M=1,N,H,W,C]

            slow_clips = self.adapt_transform(slow_clips)  # Tensor[M=1,T*C,S,S]
            slow_clips = slow_clips.view(3, self.num_segments, 3, *slow_clips.shape[-2:])

            fast_clips = self.adapt_transform(fast_clips)  # Tensor[M=1,T*C,S,S]
            fast_clips = fast_clips.view(3, self.num_segments, 3, *fast_clips.shape[-2:])
            return slow_clips, fast_clips, label
        else:
            video, audio, label = self.base_dataset.__getitem__(item)  # video should be Tensor or ndarray
            slow_clips, fast_clips = self._get_test_clips(video)  # ndarray[M=1,N,H,W,C]

            slow_clips = self.test_transform(slow_clips)  # Tensor[M=1,T*C,S,S]
            slow_clips = slow_clips.view(1, self.num_segments, 3, *slow_clips.shape[-2:])

            fast_clips = self.test_transform(fast_clips)  # Tensor[M=1,T*C,S,S]
            fast_clips = fast_clips.view(1, self.num_segments, 3, *fast_clips.shape[-2:])
            return slow_clips, fast_clips, label

    def __temporal_augment(self, video):
        if isinstance(video, torch.Tensor):
            video = video.numpy().astype(np.uint8)
        video_length = video.shape[0]
        average_duration = video_length // self.num_segments
        if average_duration > 0:
            offsets0 = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif video_length > self.num_segments:
            offsets0 = np.sort(randint(video_length, size=self.num_segments))
        else:
            offsets0 = np.zeros((self.num_segments,), dtype=np.int)
        slow_clips = np.stack([video[offsets0[:self.num_segments]]])

        if video_length < self.num_segments:
            fast_clips = slow_clips.copy()
        else:
            random1 = random.randint(0, int(video_length - self.num_segments))
            offsets1 = np.array([int(random1 + x) for x in range(self.num_segments)])
            fast_clips = np.stack([video[offsets1[:self.num_segments]]])

        return slow_clips, fast_clips  # ndarray[1,N,H,W,C]

    def _get_test_clips(self, video):
        """
        取中间
        :param video:
        :return:
        """
        if isinstance(video, torch.Tensor):
            video = video.numpy().astype(np.uint8)
        video_length = video.shape[0]

        if video_length >= self.num_segments:
            tick = video_length / float(self.num_segments)
            offsets0 = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets0 = np.zeros((self.num_segments,), dtype=np.int)
        slow_clips = np.stack([video[offsets0[:self.num_segments]]])

        t_stride = 1
        sample_pos = max(1, 1 + video_length - t_stride * self.num_segments)
        start_idx = sample_pos // 2
        offsets1 = [(idx * t_stride + start_idx) % video_length for idx in range(self.num_segments)]
        fast_clips = np.stack([video[offsets1[:self.num_segments]]])

        return slow_clips, fast_clips  # ndarray[1,N,H,W,C]


class SampleFrames(object):
    def __init__(self, clip_len, frame_interval, num_clips):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def __call__(self, video, **kwargs):
        if isinstance(video, torch.Tensor):
            video = video.numpy()  # ndarray[t,h,w,c]
        num_frames = video.shape[0]
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        clip_offsets  = clip_offsets[:, None] + np.arange(self.clip_len)[None, :] * self.frame_interval
        clip_offsets = np.concatenate(clip_offsets)  # Tesnor[num_clips,clip_len]
        clip_offsets = np.mod(clip_offsets, num_frames)  # loop
        clip_offsets = clip_offsets.reshape(self.num_clips*self.clip_len)
        return video[clip_offsets]  # ndarray[num_clips,clip_len,h,w,c]


class UniformSampleFrames(object):
    def __init__(self, num_clips, clip_len, mode='equidistant'):
        self.num_clips = num_clips
        self.clip_len = clip_len
        assert mode in ['equidistant', 'random']
        self.mode = mode

    def __call__(self, video, **kwargs):
        if isinstance(video, torch.Tensor):
            video = video.numpy()  # ndarray[t,h,w,c]
        num_frames = video.shape[0]
        if self.mode == 'equidistant':
            if num_frames < self.clip_len:
                clip_offsets = []
                start = np.random.randint(0, num_frames, size=self.num_clips)
                for st in start:
                    clip_offsets.append(np.arange(st, st + self.clip_len))
                clip_offsets = np.concatenate(clip_offsets)
            elif self.clip_len <= num_frames < 2 * self.clip_len:
                clip_offsets = []
                for _ in range(self.num_clips):
                    basic = np.arange(self.clip_len)
                    inds = np.random.choice(self.clip_len + 1, num_frames - self.clip_len, replace=False)
                    offset = np.zeros(self.clip_len + 1, dtype=np.int64)
                    offset[inds] = 1
                    offset = np.cumsum(offset)
                    clip_offsets.append(basic + offset[:-1])
                clip_offsets = np.concatenate(clip_offsets)
            else:
                segment_len = num_frames // self.clip_len
                interval = np.random.randint(low=0, high=segment_len, size=self.num_clips)
                interval = interval.repeat(self.clip_len)  # ndarray(self.num_clips*self.clip_len)
                clip_offsets = np.arange(self.clip_len) * segment_len
                clip_offsets = clip_offsets.reshape(1, -1)
                clip_offsets = clip_offsets.repeat(self.num_clips, axis=0)
                clip_offsets = clip_offsets.reshape(-1)
                clip_offsets = clip_offsets + interval
            clip_offsets = np.mod(clip_offsets, num_frames)  # loop
            return video[clip_offsets]  # ndarray[num_clips,clip_len,h,w,c]
        else:
            pass


class ClipToPIL(object):
    def __init__(self, num_clips, clip_len):
        self.num_clips = num_clips
        self.clip_len = clip_len

    def __call__(self, clips, **kwargs):
        clips = clips.reshape(self.num_clips*self.clip_len, *clips.shape[-3:])
        return [Image.fromarray(np.uint8(img)) for img in clips]


