# AME
![Update](https://img.shields.io/github/last-commit/Alvin-Zeng/AME?color=green&label=last-updated&logo=update&style=flat-squre) [![Contributor](https://img.shields.io/static/v1?label=by&message=Lujule&color=blue&style=flat-squre)](https://github.com/Lujule)
 

This technique reduces the adaptation difficulty often caused by poor performance on out-of-distribution test data before adaptation.

A curated list of temporal action localization/detection and related area (e.g. temporal action proposal) resources.

---

Contributors: Runhao Zeng, Qi Deng, Huixuan Xu, Shuaicheng Niu, Jian Chen

### <span id = "Use guide"> **Usage Guide** </span>
## <span id = "Prerequisites"> **Prerequisites** </span>
The training and testing in AME is reimplemented in PyTorch for the ease of use.
- [PyTorch](https://github.com/pytorch/pytorch)
Other minor Python modules can be installed by running
```shell
pip install -r requirements.txt
```
### <span id = "Code and Data Preparation"> **Code and Data Preparation** </span>
## <span id = "Get the code"> **Get the code** </span>
```shell
git clone --recursive https://github.com/Alvin-Zeng/AME
```

## <span id = "tal"> **Dataset** </span>
- [UCF101_noise](https://files.icg.tugraz.at/d/3551df694e3d4d6b89da/?p=%2Fucf_corrupted_videos&mode=list)
- [UCF101_contrast](https://drive.google.com/file/d/13QHfzMlu8Vjoo6NtmJGiI_whejyKTCKK/view?usp=drive_link)

### <span id = "Training AME"> **Training and Testing AME** </span>
You can use this command to training and testing AME  
You can use a for loop iterates to control which noise you want to train and test AME.If you want to change other noise,you can replace contrast with the name of another noise.
```shell
for CORRUPT in contrast 
do
    CUDA_VISIBLE_DEVICES=7 python main.py \
    --seed=507 \
    --log_dir=log_seed/tanet_ucf101/${CORRUPT}/5e-6 \
    --time_log \
    --dataset=ucf101-${CORRUPT} \
    --checkpoint=$PATH_OF_TRAINING_CHECKPOINT \
    --dataset= $PATH_OF_TRAINING_DATASET \
    --save_ckpt= $PATH_OF_SAVING_TRAINING_CHECKPOINT \
    --mix \
    --lr=5e-6 \
    --gpus 0
done
```

You can also use other commands in the script folder to train different dataset


## <span id = "tanet-tab"> **Comparisons of test-time adaptation performance on UCF101 dataset. * video domain adaptation method.** </span>
|                     Method                      |  gauss  |   pepper   |   salt   |   shot   |   zoom   |   impulse   |   motion     |   jpeg      |   contrast  |     rain    |     h265.abr     |     avg     |
| :---------------------------------------------: | :-----------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| Without Adaptation | 17.50 | 23.05 | 6.85 | 71.82 | 75.55 | 16.94 | 54.77 | 82.92 | 62.89 | 81.31 | 78.54 | 51.98 |
| BN Adaptation   | 37.01 | 33.49 | 20.64 | 80.01 | 76.13 | 37.59 | 54.46 | 83.08 | 69.13 | 85.85 | 76.90 | 59.57 |
| NORM            | 41.79 | 39.70 | 22.26 | 84.54 | 80.63 | 43.38 | 61.55 | 88.00 | 70.82 | 89.29 | 80.97 | 63.90 |
| Contrast TTA    | 36.58 | 27.57 | 21.33 | 74.31 | 69.79 | 36.11 | 49.48 | 80.23 | 24.48 | 78.46 | 74.60 | 52.09 |
| SAR             | 48.48 | 43.00 | 22.60 | 85.30 | 68.60 | 35.40 | 40.43 | 86.41 | 64.93 | 81.55 | 77.39 | 59.46 |
| ATCoN*          | 60.19 | 50.60 | 32.60 | 84.80 | 78.80 | 62.50 | 69.40 | 84.70 | 71.10 | 86.30 | 78.30 | 69.03 |
| <b>Ours</b>     | 72.06 | 64.45 | 53.50 | 86.84 | 77.80 | 67.09 | 63.57 | 88.94 | 71.76 | 90.50 | 80.89 | 74.31 |

## <span id ="Citation"> **Citation**</span>
Please cite the following paper if you feel AME useful to your research
```bibtex
@inproceedings{ACMMM-AME,
  author    = {Runhao Zeng and
               Qi Deng and
               Huixuan Xu and
               Shuaicheng Niu and
               Jian Chen},
  title     = {Exploring Motion Cues for Video Test-Time Adaptation},
  booktitle   = {ACM MM2023},
  year      = {2023},
}
```
## <span id ="Contact"> **Contact**</span>
For any question, please file an issue or contact
```email
Runhao Zeng: runhaozeng.cs@gmail.com
```

