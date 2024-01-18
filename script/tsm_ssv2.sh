conda activate dq_tta
cd /mnt/cephfs/home/alvin/dengqi/TTA/AME

# contrast gaussian_noise h265_abr jpeg_compression rain salt_noise zoom_blur motion_blur pepper_noise impulse_noise shot_noise defocus_blur
for CORRUPT in contrast gaussian_noise
do
  CUDA_VISIBLE_DEVICES=7 python main.py \
  --seed=507 \
  --log_dir=log_ssv2/tsm_ssv2/${CORRUPT} \
  --time_log \
  --model=tsm --head=tsm \
  --dataset=ssv2 \
  --corrupt_type=${CORRUPT} \
  --save_ckpt=tta_ckpt_ssv2/tsm_ssv2/${CORRUPT} \
  --mix \
  --lr=5e-6 \
  --gpus 0
done