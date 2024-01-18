conda activate dq_tta
cd /mnt/cephfs/home/alvin/dengqi/TTA/AME

# contrast gaussian_noise h265_abr jpeg_compression rain salt_noise zoom_blur motion_blur pepper_noise impulse_noise shot_noise defocus_blur
for CORRUPT in motion_blur
do
  CUDA_VISIBLE_DEVICES=1 python main.py \
  --seed=507 \
  --log_dir=log_ssv2_corrupt_online/tanet_ssv2/${CORRUPT} \
  --time_log \
  --dataset=ssv2 \
  --corrupt_type=${CORRUPT} \
  --save_ckpt=tta_ckpt_ssv2_corrupt_online/tanet_ssv2/${CORRUPT} \
  --mix \
  --lr=5e-7 \
  --gpus 0
done

for CORRUPT in zoom_blur
do
  for SEED in 2023
  do
    CUDA_VISIBLE_DEVICES=3 python main.py \
    --seed=${SEED} \
    --log_dir=log_ssv2_corrupt_online_zoom/tanet_ssv2/${CORRUPT}/${SEED} \
    --time_log \
    --dataset=ssv2 \
    --corrupt_type=${CORRUPT} \
    --save_ckpt=tta_ckpt_ssv2_corrupt_online_zoom/tanet_ssv2/${CORRUPT}/${SEED} \
    --mix \
    --lr=5e-7 \
    --gpus 0
  done
done
