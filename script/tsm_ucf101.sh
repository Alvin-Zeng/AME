conda activate dq_tta
cd /mnt/cephfs/home/alvin/dengqi/TTA/AME

# gauss pepper salt shot zoom impulse defocus motion jpeg contrast rain h265_abr
for CORRUPT in jpeg contrast rain h265_abr
do
  CUDA_VISIBLE_DEVICES=3 python main.py \
  --seed=507 \
  --log_dir=log/tsm_ucf101/${CORRUPT} \
  --model=tsm --head=tsm \
  --checkpoint=/mnt/cephfs/home/alvin/dengqi/TTA/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar \
  --time_log \
  --dataset=ucf101-${CORRUPT} \
  --save_ckpt=tta_ckpt/tsm_ucf101/${CORRUPT} \
  --mix \
  --lr=5e-5 \
  --gpus 0
done

for CORRUPT in salt_ours
do
  CUDA_VISIBLE_DEVICES=7 python main.py \
  --seed=507 \
  --log_dir=log_ours/tsm_ucf101/${CORRUPT} \
  --model=tsm --head=tsm \
  --checkpoint=/mnt/cephfs/home/alvin/dengqi/TTA/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar \
  --time_log \
  --dataset=ucf101 --corrupt_type='salt_noise' \
  --save_ckpt=tta_ckpt_ours/tsm_ucf101/${CORRUPT} \
  --mix \
  --lr=5e-5 \
  --gpus 0
done
