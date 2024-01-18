conda activate dq_tta
cd /mnt/cephfs/home/alvin/dengqi/TTA/AME

# gauss pepper salt shot zoom impulse defocus motion jpeg contrast rain h265_abr
for CORRUPT in defocus motion
do
  CUDA_VISIBLE_DEVICES=4 python main.py \
  --seed=507 \
  --log_dir=log/tanet_ucf101/${CORRUPT} \
  --time_log \
  --dataset=ucf101-${CORRUPT} \
  --save_ckpt=tta_ckpt/tanet_ucf101/${CORRUPT} \
  --mix \
  --lr=5e-4 \
  --gpus 0
done

for CORRUPT in salt_ours
do
  CUDA_VISIBLE_DEVICES=5 python main.py \
  --seed=507 \
  --log_dir=log_ours/tanet_ucf101/${CORRUPT} \
  --time_log \
  --dataset=ucf101 --corrupt_type='salt_noise' \
  --save_ckpt=tta_ckpt_ours/tanet_ucf101/${CORRUPT} \
  --mix \
  --lr=5e-4 \
  --gpus 0
done

# 修改seed重复测试zoom h265_abr
for CORRUPT in h265_abr
do
  for SEED in 507
  do
    CUDA_VISIBLE_DEVICES=7 python main.py \
    --seed=${SEED} \
    --log_dir=log_seed/tanet_ucf101/${CORRUPT}/5e-6/${SEED} \
    --time_log \
    --dataset=ucf101-${CORRUPT} \
    --save_ckpt=tta_ckpt_seed/tanet_ucf101/${CORRUPT}/5e-6/${SEED} \
    --mix \
    --lr=5e-6 \
    --gpus 0
  done
done