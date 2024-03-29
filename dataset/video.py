import tqdm
import torch
import bisect
import math
import warnings
from fractions import Fraction
from typing import List
from torchvision.io import (
    _probe_video_from_file,
    _read_video_from_file,
    read_video,
    read_video_timestamps,
)

from torchvision.datasets.video_utils import pts_convert, unfold, _VideoTimestampsDataset, _collate_fn


class VideoClips(object):
    def __init__(
        self,
        video_paths,
        clip_length_in_frames=16,
        frames_between_clips=1,
        frame_rate=None,
        _precomputed_metadata=None,
        num_workers=0,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _video_max_dimension=0,
        _audio_samples=0,
        _audio_channels=0,
    ):

        self.video_paths = video_paths
        self.num_workers = num_workers

        # these options are not valid for pyav backend
        self._video_width = _video_width
        self._video_height = _video_height
        self._video_min_dimension = _video_min_dimension
        self._video_max_dimension = _video_max_dimension
        self._audio_samples = _audio_samples
        self._audio_channels = _audio_channels

        if _precomputed_metadata is None:
            self._compute_frame_pts()
        else:
            self._init_from_metadata(_precomputed_metadata)
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)

    def _compute_frame_pts(self):
        self.video_pts = []
        self.video_fps = []

        # strategy: use a DataLoader to parallelize read_video_timestamps
        # so need to create a dummy dataset first
        import torch.utils.data

        dl = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(self.video_paths),
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

        with tqdm.tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps = list(zip(*batch))
                # we need to specify dtype=torch.long because for empty list,
                # torch.as_tensor will use torch.float as default dtype. This
                # happens when decoding fails and no pts is returned in the list.
                clips = [torch.as_tensor(c, dtype=torch.long) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)

    def _init_from_metadata(self, metadata):
        self.video_paths = metadata["video_paths"]
        assert len(self.video_paths) == len(metadata["video_pts"])
        self.video_pts = metadata["video_pts"]
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_fps = metadata["video_fps"]

    @property
    def metadata(self):
        _metadata = {
            "video_paths": self.video_paths,
            "video_pts": self.video_pts,
            "video_fps": self.video_fps,
        }
        return _metadata

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        video_fps = [self.video_fps[i] for i in indices]
        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        return type(self)(
            video_paths,
            self.num_frames,
            self.step,
            self.frame_rate,
            _precomputed_metadata=metadata,
            num_workers=self.num_workers,
            _video_width=self._video_width,
            _video_height=self._video_height,
            _video_min_dimension=self._video_min_dimension,
            _video_max_dimension=self._video_max_dimension,
            _audio_samples=self._audio_samples,
            _audio_channels=self._audio_channels,
        )

    @staticmethod
    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(
            int(math.floor(total_frames)), fps, frame_rate
        )
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if not clips.numel():
            warnings.warn("There aren't enough frames in the current video to get a clip for the given clip length and "
                          "frames between clips. The video (and potentially others) will be skipped.")
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs

    def compute_clips(self, num_frames, step, frame_rate=None):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.

        Args:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
            frame_rate (int, optional): The frame rate
        """
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            clips, idxs = self.compute_clips_for_video(
                video_pts, num_frames, step, fps, frame_rate
            )
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_clip(self, idx):
        """
        Gets a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        from torchvision import get_video_backend

        backend = get_video_backend()

        if backend == "pyav":
            # check for invalid options
            if self._video_width != 0:
                raise ValueError("pyav backend doesn't support _video_width != 0")
            if self._video_height != 0:
                raise ValueError("pyav backend doesn't support _video_height != 0")
            if self._video_min_dimension != 0:
                raise ValueError(
                    "pyav backend doesn't support _video_min_dimension != 0"
                )
            if self._video_max_dimension != 0:
                raise ValueError(
                    "pyav backend doesn't support _video_max_dimension != 0"
                )
            if self._audio_samples != 0:
                raise ValueError("pyav backend doesn't support _audio_samples != 0")

        if backend == "pyav":
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            video, audio, info = read_video(video_path, start_pts, end_pts)
        else:
            info = _probe_video_from_file(video_path)
            video_fps = info.video_fps
            audio_fps = None

            video_start_pts = clip_pts[0].item()
            video_end_pts = clip_pts[-1].item()

            audio_start_pts, audio_end_pts = 0, -1
            audio_timebase = Fraction(0, 1)
            video_timebase = Fraction(
                info.video_timebase.numerator, info.video_timebase.denominator
            )
            if info.has_audio:
                audio_timebase = Fraction(
                    info.audio_timebase.numerator, info.audio_timebase.denominator
                )
                audio_start_pts = pts_convert(
                    video_start_pts, video_timebase, audio_timebase, math.floor
                )
                audio_end_pts = pts_convert(
                    video_end_pts, video_timebase, audio_timebase, math.ceil
                )
                audio_fps = info.audio_sample_rate
            video, audio, info = _read_video_from_file(
                video_path,
                video_width=self._video_width,
                video_height=self._video_height,
                video_min_dimension=self._video_min_dimension,
                video_max_dimension=self._video_max_dimension,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase=video_timebase,
                audio_samples=self._audio_samples,
                audio_channels=self._audio_channels,
                audio_pts_range=(audio_start_pts, audio_end_pts),
                audio_timebase=audio_timebase,
            )

            info = {"video_fps": video_fps}
            if audio_fps is not None:
                info["audio_fps"] = audio_fps

        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate
        # There is Some Bug. Hence, I Rewrite It.
        # assert len(video) == self.num_frames, "{} x {}".format(
        #     video.shape, self.num_frames
        # )
        return video[:self.num_frames], audio[:self.num_frames], info, video_idx

    def get_video(self, video_idx):  # 获取视频本身
        from torchvision import get_video_backend
        backend = get_video_backend()

        video_path = self.video_paths[video_idx]

        if backend == "pyav":
            # check for invalid options
            if self._video_width != 0:
                raise ValueError("pyav backend doesn't support _video_width != 0")
            if self._video_height != 0:
                raise ValueError("pyav backend doesn't support _video_height != 0")
            if self._video_min_dimension != 0:
                raise ValueError("pyav backend doesn't support _video_min_dimension != 0")
            if self._video_max_dimension != 0:
                raise ValueError("pyav backend doesn't support _video_max_dimension != 0")
            if self._audio_samples != 0:
                raise ValueError("pyav backend doesn't support _audio_samples != 0")

        if backend == "pyav":
            video, audio, info = read_video(video_path)
        else:
            pass

        return video, audio, info

    def __getstate__(self):
        video_pts_sizes = [len(v) for v in self.video_pts]
        # To be back-compatible, we convert data to dtype torch.long as needed
        # because for empty list, in legacy implementation, torch.as_tensor will
        # use torch.float as default dtype. This happens when decoding fails and
        # no pts is returned in the list.
        video_pts = [x.to(torch.int64) for x in self.video_pts]
        # video_pts can be an empty list if no frames have been decoded
        if video_pts:
            video_pts = torch.cat(video_pts)
            # avoid bug in https://github.com/pytorch/pytorch/issues/32351
            video_pts = video_pts.numpy()

        # make a copy of the fields of self
        d = self.__dict__.copy()
        d["video_pts_sizes"] = video_pts_sizes
        d["video_pts"] = video_pts
        # delete the following attributes to reduce the size of dictionary. They
        # will be re-computed in "__setstate__()"
        del d["clips"]
        del d["resampling_idxs"]
        del d["cumulative_sizes"]

        # for backwards-compatibility
        d["_version"] = 2
        return d

    def __setstate__(self, d):
        # for backwards-compatibility
        if "_version" not in d:
            self.__dict__ = d
            return

        video_pts = torch.as_tensor(d["video_pts"], dtype=torch.int64)
        video_pts = torch.split(video_pts, d["video_pts_sizes"], dim=0)
        # don't need this info anymore
        del d["video_pts_sizes"]

        d["video_pts"] = video_pts
        self.__dict__ = d
        # recompute attributes "clips", "resampling_idxs" and other derivative ones
        self.compute_clips(self.num_frames, self.step, self.frame_rate)