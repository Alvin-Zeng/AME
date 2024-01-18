import os.path


def init_config(args):
    # for all
    if 'ucf101' in args.dataset:
        if args.dataset == 'ucf101':
            args.dataset_path = "/mnt/cephfs/dataset/m3lab_data/dengqi/UCF101/UCF-101"
            args.classes_path = "/mnt/cephfs/dataset/m3lab_data/dengqi/UCF101/ucfTrainTestlist/classInd.txt"
            args.annotation = "/mnt/cephfs/dataset/m3lab_data/dengqi/UCF101/ucfTrainTestlist"
            args.extension = '.avi'
        else:
            corrupt_type = args.dataset.replace('ucf101-', '')
            corrupt_dir = "/mnt/cephfs/dataset/m3lab_data/dengqi/UCF101_corrupt/level_5_ucf_val_split_1_"

            args.corrupt_type = 'origin'
            args.dataset_path = os.path.join(corrupt_dir, corrupt_type)
            args.classes_path = "/mnt/cephfs/dataset/m3lab_data/dengqi/UCF101/ucfTrainTestlist/classInd.txt"
            args.annotation = "/mnt/cephfs/dataset/m3lab_data/dengqi/UCF101/ucfTrainTestlist"
            args.extension = '.mp4'

        if args.model == 'tanet':
            args.checkpoint = "/mnt/cephfs/home/alvin/dengqi/TTA/checkpoint/TANet_UCF.pth.tar"
        elif args.model == 'tsm':
            args.checkpoint = "/mnt/cephfs/home/alvin/dengqi/TTA/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar"
        else:
            raise NotImplementedError

    elif args.dataset == 'ssv2':
        if args.model == 'tanet':
            args.checkpoint = "/mnt/cephfs/home/alvin/dengqi/TTA/checkpoint/tanet_ssv2_256x256.pth.tar"
        elif args.model == 'tsm':
            args.checkpoint = "/mnt/cephfs/home/alvin/dengqi/TTA/checkpoint/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth"
        
        # args.dataset_path = "/mnt/cephfs/dataset/m3lab_data/dengqi/SSV2/20bn-something-something-v2"
        args.dataset_path = "/mnt/cephfs/dataset/smth-smth-v2/20bn-something-something-v2"
        args.classes_path = "/mnt/cephfs/dataset/m3lab_data/dengqi/SSV2/annotations/something-something-v2-labels.json"
        args.annotation = "/mnt/cephfs/dataset/m3lab_data/dengqi/SSV2/annotations/something-something-v2-validation.json"
        args.corrupt_dir = "/mnt/cephfs/dataset/m3lab_data/dengqi/SSV2_C"
        
        args.is_corrupted = True
        # if args.corrupt_type in ['gaussian_noise', 'contrast', 'rain', 'salt_noise', 'zoom_blur', 'h265_abr']:
        #     args.is_corrupted = True
        # else:
        #     args.is_corrupted = False

    # for experiment
    args.shuffle = True
    args.negative  = not args.cross_entropy
    args.alignment = True
    args.alignment_half = True
    args.mix = True