import json
import logging
import pathlib
import random
import argparse
import copy
import time
import os
import zipfile
from glob import glob
from urllib.parse import quote
import sys
import socket
from fnmatch import fnmatch, fnmatchcase
import gc
import shutil

import numpy as np
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader


from model.recognizer import build_recognizer
from model.combination import Combination, CombinationMoco
from dataset.corrupt import ffmpeg_corrupt_dict
from dataset.ucf101 import UCF101
from dataset.ssv2 import SSV2, SSV2C
from dataset.data import SlowFastDataset
from tta import eval_and_label_dataset, train_epoch, eval_dataset_mixed
from config import init_config

import warnings
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="TTA")
# seed
parser.add_argument('--seed', type=int, default=507)
# gpus
parser.add_argument('--gpus', nargs='+', type=int, default=None)
# log args
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--time_log', default=False, action='store_true')
parser.add_argument('--report_frequence', type=int, default=100)
parser.add_argument('--aux_log', default=False,action='store_true')
parser.add_argument('--save_ckpt', type=str, default=None)
# model args
parser.add_argument('--model', type=str, default='tanet', choices=['tanet', 'swin', 'tsm'])
parser.add_argument('--head', type=str, default='tsm', choices=['tsn', 'tsm', 'i3d'])
parser.add_argument('--checkpoint', type=str, default="/mnt/cephfs/home/alvin/dengqi/TTA/checkpoint/TANet_UCF.pth.tar")
parser.add_argument('--num_segments', type=int, default=16)
parser.add_argument('--shared_layers', type=int, default=4)  
parser.add_argument('--deep_mult', type=int, default=1)
parser.add_argument('--K', type=int, default=400)
parser.add_argument('--m', type=float, default=0.999)
parser.add_argument('--T', type=float, default=0.07)
parser.add_argument('--refine_method', type=str, default="nearest_neighbors")
parser.add_argument('--num_neighbors', type=int, default=2)
parser.add_argument('--dist_type', type=str, default="cosine", choices=["cosine", "euclidean"])
# dataset args
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--dataset_path', type=str, default='/mnt/cephfs/dataset/UCF101/UCF-101')
parser.add_argument('--classes_path', type=str, default=None)
parser.add_argument('--annotation', type=str, default='./ucfTrainTestlist')
parser.add_argument('--dataset_workers', type=int, default=8)
parser.add_argument('--dataloader_workers', type=int, default=8)
parser.add_argument('--test_batchsize', type=int, default=16)
parser.add_argument('--train_batchsize', type=int, default=1)
parser.add_argument('--corrupt_type', type=str, default='origin', choices=['origin','gaussian_noise','salt_noise','shot_noise','contrast','motion_blur','rain','zoom_blur','impulse_noise','defocus_blur','jpeg_compression','pepper_noise', 'h265_abr'])
parser.add_argument('--corrupt_severity', type=int, default=5)
parser.add_argument('--corrupt_dir', type=str)
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--is_corrupted', default=False, action='store_true')
# optimizer args
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
# training args
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--contrast_type', type=str, default="class_aware")
# testing args
parser.add_argument('--mix', default=False, action='store_true')
# loss args
parser.add_argument('--ce_sup_type', type=str, default='weak_strong', choices=['weak_weak', 'weak_strong'])
parser.add_argument('--cross_entropy', default=False, action='store_true')
parser.add_argument('--negative', default=False, action='store_true')
parser.add_argument('--pure_random', default=False, action='store_true')
parser.add_argument('--alignment', default=False, action='store_true')
parser.add_argument('--alignment_half', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=2.0)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--alm', type=float, default=0.1)
# debug
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--source_only', default=False, action='store_true')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_sh_n_codes(path, ignore_dir=[], ignore_file=[]):
    """
    :param path:
    :param ignore_dir: 在根目录下需要被忽略的子目录或文件名或通配符
    :param ignore_file: 所有文件和目录中需要被忽略的文件名或通配符
    :return:
    """
    name = os.path.join(path, 'run_{}.sh'.format(socket.gethostname()))
    with open(name, 'w') as f:
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(str(arg) for arg in sys.argv) + '\n')

    name = os.path.join(path, 'code.zip')
    with zipfile.ZipFile(name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:

        def is_ignored(file_name, ignore_list):
            if isinstance(file_name, str):
                for i in ignore_list:
                    if fnmatchcase(file_name, i):
                        return True
                return False
            elif isinstance(file_name, list):
                for i in file_name:
                    for j in ignore_list:
                        if fnmatchcase(i, j):
                            return True
                return False
            else:
                raise NotImplementedError

        first_list = glob('*', recursive=True)
        first_list = [i for i in first_list if not is_ignored(i, ignore_dir)]

        file_list = []
        patterns = [x + '/**' for x in first_list]
        for pattern in patterns:
            file_list.extend(glob(pattern, recursive=True))

        file_list = [x[:-1] if x[-1] == "/" else x for x in file_list]
        for filename in file_list:
            if not is_ignored(filename.split('/'), ignore_file):
                zf.write(filename)


class Log:
    def __init__(self, main_file, aux_file):
        self.main_file = pathlib.Path(main_file)
        self.aux_file = pathlib.Path(aux_file)
        self.main_log = open(self.main_file, 'a')
        self.aux_log = open(self.aux_file, 'a')

    def write(self, message):
        try:
            self.aux_log.write(message)
            self.main_log.write(message)
        except OSError as err:
            print(err)
        except:
            print(sys.exc_info())

    def flush(self):
        try:
            self.aux_log.flush()
            self.main_log.flush()
        except OSError as err:
            print(err)
        except:
            print(sys.exc_info())

    def close(self):
        self.aux_log.close()
        self.main_log.close()


class Logger:
    def __init__(self, main_path, aux_path=None):
        # create loggers
        self.main_logger = logging.getLogger()
        self.main_logger.setLevel(logging.INFO)
        # file handler
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        main_handler = logging.FileHandler(main_path / 'tta.log', mode='a')
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(fmt)
        if aux_path is not None:
            aux_handler = logging.FileHandler(aux_path / 'tta.log', mode='a')
            aux_handler.setLevel(logging.INFO)
            aux_handler.setFormatter(fmt)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        # add handler
        self.main_logger.addHandler(main_handler)
        if aux_path is not None:
            self.main_logger.addHandler(aux_handler)
        self.main_logger.addHandler(stream_handler)

    def info(self, message):
        try:
            self.main_logger.info(message)
        except:
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(exc_type)
            print(exc_value)
            print(exc_tb)


def init_logger():
    if args.time_log:
        now_time = time.localtime()
        log_dir = pathlib.Path(args.log_dir) / f'{now_time.tm_mon}-{now_time.tm_mday}-{now_time.tm_hour}:{now_time.tm_min}'
        if log_dir.exists():
            log_dir = pathlib.Path(args.log_dir) / f'{now_time.tm_mon}-{now_time.tm_mday}-{now_time.tm_hour}:{now_time.tm_min}:{now_time.tm_sec}'
        log_dir.mkdir(parents=True)
    else:
        log_dir = pathlib.Path(args.log_dir)
        log_dir.mkdir(parents=True)

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    ip_log = open(log_dir / f'{ip}.txt', 'w')
    ip_log.close()
    if args.aux_log:
        ckpt_dir = pathlib.Path.home() / 'tta_ckpts' / f'{args.corrupt_type}_{args.lr}' / f'{now_time.tm_mon}-{now_time.tm_mday}-{now_time.tm_hour}:{now_time.tm_min}:{now_time.tm_sec}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = None

    # save_sh_n_codes(path=log_dir,
    #                 ignore_dir=['reference', '*log', '*log*', 'script', 'Video-Corruption-Robustness'],
    #                 ignore_file=['__pycache__', '*.pth', '*.sh'])

    args_log = open(log_dir / 'args.csv', 'w')
    args_log.write("arg, value\n")
    args_vars = vars(args)
    for k, v in args_vars.items():
        k = str(k)
        v = str(v).replace(',', ' ')
        args_log.write(f"{k},{v}\n")
    args_log.write("\n")
    args_log.flush()
    args_log.close()

    if args.aux_log:
        acc_log = Log(log_dir / 'acc.csv', ckpt_dir / 'acc.csv')
        acc_log.write("slow,fast\n")
        acc_log.flush()

        loss_log = Log(log_dir / 'loss.csv', ckpt_dir / 'loss.csv')
        loss_log.write("Epoch,i,loss,loss_cls,loss_nce,loss_div,loss_alm\n")
        loss_log.flush()

    else:
        acc_log = open(log_dir / 'acc.csv', 'a')
        acc_log.write("slow,fast\n")
        acc_log.flush()

        loss_log = open(log_dir / 'loss.csv', 'a')
        loss_log.write("Epoch,i,loss,loss_cls,loss_nce,loss_div,loss_alm\n")
        loss_log.flush()
    

    logger = Logger(log_dir, ckpt_dir)

    return logger, acc_log, loss_log, ckpt_dir


def init_model(model: str, dataset:str, checkpoint: str, num_segments: int, shared_layers: int, deep_mult: int, moco_args: dict):
    if args.dataset == 'mini-ssv2':
        with open(args.classes_path, 'r') as f:
            ssv2_classes = json.load(f)
        with open(args.annotation, 'r') as f:
            mini_classes = json.load(f)['labels']
        base_model = build_recognizer(model=args.model, head=args.head, dataset=args.dataset, checkpoint=args.checkpoint,
                                      num_segments=args.num_segments, ssv2_classes=ssv2_classes, mini_classes=mini_classes)
    else:
        base_model = build_recognizer(model=model, head=args.head, dataset=dataset,
                                      checkpoint=checkpoint, num_segments=num_segments)
    model_slow = copy.deepcopy(base_model)
    model_fast = copy.deepcopy(base_model)
    momentum_model_slow = copy.deepcopy(base_model)
    momentum_model_fast = copy.deepcopy(base_model)
    if model == 'tsm':
        src_model = Combination(model_slow, model_fast, 6)
    else:
        src_model = Combination(model_slow, model_fast, shared_layers)
    momentum_model = Combination(momentum_model_slow, momentum_model_fast, shared_layers)
    K = moco_args.get('K', 400)
    m = moco_args.get('m', 0.999)
    T_moco = moco_args.get('T_moco', 0.07)
    if len(args.gpus) <= 1:
        model = CombinationMoco(src_model, momentum_model, K=K, m=m, T_moco=T_moco).cuda()
    else:
        model = CombinationMoco(src_model, momentum_model, K=K, m=m, T_moco=T_moco)
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    return model, src_model.get_optim_policies(args.lr, deep_mult)

#将两个数据集 ssv2和ucf101导入模型
def init_data():
    if 'ucf101' in args.dataset:
        dataset = UCF101(root=args.dataset_path, classes_path=args.classes_path, annotation_path=args.annotation,
                           extension=args.extension, debug=args.debug)
        dataset = SlowFastDataset(dataset, args.corrupt_type, args.corrupt_severity, args.num_segments)
    elif args.dataset == 'ssv2':
        # dataset = SSV2C(root=args.dataset_path, corrupt_dir=args.corrupt_dir,
        #                 corrupt_type=args.corrupt_type, corrupt_severity=args.corrupt_severity,
        #                 classes_path=args.classes_path, annotation_path=args.annotation,
        #                 debug=args.debug, mode=args.dataset, is_corrupted=args.is_corrupted)
        # args.corrupt_type = 'origin'
        # dataset = SlowFastDataset(dataset, args.corrupt_type, args.corrupt_severity,
        #                           args.num_segments, scale_size=288, input_size=256)
        
        dataset = SSV2(root=args.dataset_path, classes_path=args.classes_path, 
                       annotation_path=args.annotation, debug=args.debug, mode=args.dataset)
        dataset = SlowFastDataset(dataset, args.corrupt_type, args.corrupt_severity, 
                                  args.num_segments, scale_size=288, input_size=256, is_corrupted=False)
        
    elif args.dataset == 'mini-ssv2':
        if args.corrupt_type in ffmpeg_corrupt_dict:
            dataset = SSV2C(root=args.dataset_path, corrupt_dir=args.corrupt_dir,
                            corrupt_type=args.corrupt_type, corrupt_severity=args.corrupt_severity,
                            classes_path=args.classes_path, annotation_path=args.annotation,
                            debug=args.debug, mode=args.dataset)
        else:
            dataset = SSV2(root=args.dataset_path, classes_path=args.classes_path,
                           annotation_path=args.annotation, debug=args.debug, mode=args.dataset)
        is_corrupted = args.corrupt_type in ffmpeg_corrupt_dict
        dataset = SlowFastDataset(dataset, args.corrupt_type, args.corrupt_severity,
                                  args.num_segments, scale_size=288, input_size=256, is_corrupted=is_corrupted)
    else:
        raise NotImplementedError
    test_dataloader = DataLoader(dataset, batch_size=args.test_batchsize * len(args.gpus),
                                 num_workers=args.test_batchsize, shuffle=args.shuffle, drop_last=False)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batchsize * len(args.gpus),
                                  shuffle=True, num_workers=args.train_batchsize, drop_last=False)
    return test_dataloader, train_dataloader


def tta():
    init_config(args)

    logger, acc_log, loss_log, ckpt_dir = init_logger()

    logger.info("1 - Created target model")
    moco_args = {'K': args.K, 'm': args.m, 'T_moco': args.T}
    #模型的构建
    model, params = init_model(args.model, args.dataset, args.checkpoint, args.num_segments, args.shared_layers, args.deep_mult, moco_args)
    #数据导入
    logger.info("2 - Created dataloader")
    test_loader, train_loader = init_data()
    #计算初始准确率
    logger.info("3 - Computed initial acc")
    if args.mix:
        eval_dataset_mixed(test_loader,        model, logger, acc_log, epoch=-1)
    else:
        eval_and_label_dataset(dataloader=test_loader, model=model, sample_type="slow",
                                logger=logger, acc_log=acc_log, epoch=-1, args=args)
        eval_and_label_dataset(dataloader=test_loader, model=model, sample_type="fast",
                                logger=logger, acc_log=acc_log, epoch=-1, args=args)
                

    logger.info("4 - Created optimizer")
    #梯度下降
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    logger.info("Start training...")
    #
    for epoch in range(args.epochs):
        # train for one epoch
        if args.mix:
            train_epoch(train_loader=train_loader, model=model,
                        optimizer=optimizer, logger=logger, loss_log=loss_log, epoch=epoch, args=args)
            eval_dataset_mixed(test_loader, model, logger, acc_log, epoch)
        else:
            train_epoch(train_loader=train_loader, model=model,
                        optimizer=optimizer, logger=logger, loss_log=loss_log, epoch=epoch, args=args)
            eval_and_label_dataset(dataloader=test_loader, model=model, sample_type="slow",
                                    logger=logger, acc_log=acc_log, epoch=epoch, args=args)
            eval_and_label_dataset(dataloader=test_loader, model=model, sample_type="fast",
                                    logger=logger, acc_log=acc_log, epoch=epoch, args=args)
        
        if args.save_ckpt is not None:
            logger.info(f"Epoch [{epoch}] Save Checkpoint")
            # torch.save({'state_dict': model.state_dict(), "params":params}, args.save_ckpt / f'ckpt{epoch}.pth')
            save_dir = pathlib.Path(args.save_ckpt)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"checkpoint_{epoch:04d}.pth.tar"
            save_path = os.path.join(args.save_ckpt, filename)
            save_checkpoint(model, optimizer, epoch, save_path=save_path)


def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth.tar"):
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_path)


if __name__ == "__main__":
    set_seed(args.seed)
    tta()
