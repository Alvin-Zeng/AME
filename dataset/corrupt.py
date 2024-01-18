import json
import subprocess

import numpy as np
from PIL import Image
import torch
import cv2
from scipy.ndimage import zoom as scizoom
import skimage
from io import BytesIO
from scipy import ndimage, misc
import random


class origin:
    def __init__(self, severity=None):
        pass
    def __call__(self, x):
        if isinstance(x, Image.Image):
            return x
        elif isinstance(x, np.ndarray):
            return Image.fromarray(np.uint8(x))


class gaussian_noise:
    def __init__(self, severity):
        self.c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        

    def __call__(self, x):
        x = np.array(x) / 255.
        return Image.fromarray(np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=self.c), 0, 1) * 255))


class pepper_noise:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, x):
        im1 = np.array(x)
        mask = np.random.randint(0, 100, im1.shape)
        im2 = np.where(mask < self.severity * 10 , 0, im1)
        return  Image.fromarray(np.uint8(np.clip(im2, 0, 255)))


class salt_noise:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, x):
        im1 = np.array(x)
        mask = np.random.randint(0, 100, im1.shape)
        im2 = np.where(mask < self.severity * 10 , 255, im1)
        return  Image.fromarray(np.uint8(np.clip(im2, 0, 255)))


class shot_noise:
    def __init__(self, severity):
        self.c = [60, 25, 12, 5, 3][severity - 1]

    def __call__(self, x):
        x = np.array(x) / 255.
        # return np.clip(np.random.poisson(x * self.c) / float(self.c), 0, 1) * 255
        return Image.fromarray(np.uint8(np.clip(np.random.poisson(x * self.c) / float(self.c), 0, 1) * 255))


class zoom_blur:
    def __init__(self, severity):
        self.c = [np.arange(1, 1.11, 0.01),
                  np.arange(1, 1.16, 0.01),
                  np.arange(1, 1.21, 0.02),
                  np.arange(1, 1.26, 0.02),
                  np.arange(1, 1.31, 0.03)][severity - 1]

    def __call__(self, x):
        x = (np.array(x) / 255.0).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in self.c:
            out += self.clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(self.c) + 1)
        return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255.0))

    @staticmethod
    def clipped_zoom(img, zoom_factor):
        """

        :param img: Tensor[h,w,c]
        :param zoom_factor:
        :return:
        """
        h = img.shape[0]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / zoom_factor))
        top = (h - ch) // 2
        img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2
        return img[trim_top:trim_top + h, trim_top:trim_top + h]


class impulse_noise:
    def __init__(self, severity):
        self.c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    def __call__(self, x):
        x = skimage.util.random_noise(np.array(x) / 255., mode='s&p', amount=self.c)
        return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))


class defocus_blur:
    def __init__(self, severity):
        self.c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    def __call__(self, x):
        x = np.array(x) / 255.0
        kernel = self.disk(radius=self.c[0], alias_blur=self.c[1])
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return Image.fromarray(np.uint8(np.clip(channels, 0, 1) * 255))

    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class motion_blur:
    def __init__(self, severity):
        # motion_overlapping_frames = [1,2,3,4,6]
        motion_overlapping_frames = [3, 5, 7, 9, 11]
        self.c = motion_overlapping_frames[severity-1]

    def __call__(self, x):
        clip=np.asarray(x)
        blur_clip=[]
        for i in range(self.c, clip.shape[0] - self.c):
            blur_image=np.sum(clip[i - self.c : i + self.c],axis=0,dtype=np.float)/(2.0 * self.c)
            blur_clip.append(np.array(blur_image,dtype=np.uint8))
        # return blur_clip
        return Image.fromarray(np.uint8(blur_clip))


class jpeg_compression:
    def __init__(self, severity):
        self.c = [25, 18, 15, 10, 7][severity - 1]

    def __call__(self, x):
        output = BytesIO()
        x.save(output, 'JPEG', quality=self.c)
        x = Image.open(output)

        return x


class contrast:
    def __init__(self, severity):
        # self.c = [0.5, 0.4, .3, .2, .1][severity - 1]
        self.c = [0.4, .3, .2, .1, .05][severity - 1]

    def __call__(self, x):
        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return Image.fromarray(np.uint8(np.clip((x - means) * self.c + means, 0, 1) * 255))    


class rain:
    def __init__(self, severity):
        self.severity = severity
        
    def __call__(self, image):
        image = np.asarray(image)
        slant = -1
        drop_length = 20
        drop_width = 1
        drop_color = (220, 220, 220)
        rain_type = self.severity
        darken_coefficient = [0.8, 0.8, 0.7, 0.6, 0.5]
        slant_extreme = slant

        imshape = image.shape
        if slant_extreme == -1:
            slant = np.random.randint(-10, 10)  ##generate random slant if no slant value is given
        rain_drops, drop_length = self.generate_random_lines(imshape, slant, drop_length, rain_type)
        output = self.rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops,
                            darken_coefficient[self.severity - 1])
        image_RGB = output

        # return image_RGB
        return Image.fromarray(np.uint8(image_RGB)) 
    
    def generate_random_lines(self, imshape, slant, drop_length, rain_type):
        drops = []
        area = imshape[0] * imshape[1]
        no_of_drops = area // 600

        # if rain_type.lower()=='drizzle':

        if rain_type == 1:
            no_of_drops = area // 770
            drop_length = 10
            # print("drizzle")
        # elif rain_type.lower()=='heavy':
        elif rain_type == 2:
            no_of_drops = area // 770
            drop_length = 30
        # elif rain_type.lower()=='torrential':
        elif rain_type == 3:
            no_of_drops = area // 770
            drop_length = 60
            # print("heavy")
        elif rain_type == 4:
            no_of_drops = area // 500
            drop_length = 60
        elif rain_type == 5:
            no_of_drops = area // 400
            drop_length = 80
            # print('torrential')

        for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
            if slant < 0:
                x = np.random.randint(slant, imshape[1])
            else:
                x = np.random.randint(0, imshape[1] - slant)
            y = np.random.randint(0, imshape[0] - drop_length)
            drops.append((x, y))
        return drops, drop_length

    def rain_process(self, image, slant, drop_length, drop_color, drop_width, rain_drops, darken):
        imshape = image.shape
        rain_mask = np.zeros((imshape[0], imshape[1]))
        image_t = image.copy()
        for rain_drop in rain_drops:
            cv2.line(rain_mask, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length),
                    drop_color, drop_width)

        rain_mask = np.stack((rain_mask, rain_mask, rain_mask), axis=2)
        image_rain = image + np.array(rain_mask * (1 - image / 255.0) * (1 - np.mean(image) / 255.0), dtype=np.uint8)
        blur_rain = cv2.blur(image_rain, (3, 3))  ## rainy view are blurry
        image_RGB = np.array(blur_rain * rain_mask / 255.0 + image * (1 - rain_mask / 255.0))
        # blur_rain_mask=rain_mask
        image_RGB = np.array(image_RGB) / 255.
        means = np.mean(image_RGB, axis=(0, 1), keepdims=True)
        image_RGB = np.array(np.clip((image_RGB - means) * darken + means, 0, 1) * 255, dtype=np.uint8)

        return image_RGB


def h265_abr(src, dst, severity):
    c = [2, 4, 8, 16, 32][severity - 1]
    result = subprocess.Popen(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    data = json.load(result.stdout)

    bit_rate = str(int(float(data['format']['bit_rate']) / c))

    # return_code = subprocess.call(
    #     ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
    #      bit_rate, dst])
    ret = subprocess.call(["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec",
                           "libx265", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize", bit_rate, dst])

    return ret


ffmpeg_corrupt_dict = {'h265_abr': h265_abr}


corrupt_dict = {
    'origin': origin,
    'gaussian_noise': gaussian_noise,
    'salt_noise': salt_noise,
    'shot_noise': shot_noise,
    'contrast': contrast,
    'motion_blur': motion_blur,
    'rain': rain,
    'zoom_blur': zoom_blur,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'jpeg_compression': jpeg_compression,
    'pepper_noise': pepper_noise,
    'h265_abr': h265_abr
}


class Corrupt:
    def __init__(self, corrupt_type, corrupt_severity):
        self.corrupt = corrupt_dict[corrupt_type](corrupt_severity)

    def __call__(self, video):
        if isinstance(video, torch.Tensor):
            video = video.numpy().astype(np.uint8)  # ndarray[T,H,W,C] uint8
            for i in range(video.shape[0]):
                image = video[i]  # ndarray[H,W,C]
                video[i] = np.uint8(self.corrupt(image))  # ndarray[T,H,W,C]
            return video  # ndarray[T,H,W,C]
        if isinstance(video, list):  # list of PIL
            return [self.corrupt(img) for img in video]  # list of PIL



