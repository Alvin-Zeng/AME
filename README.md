# AME
![Update](https://img.shields.io/github/last-commit/Alvin-Zeng/AME?color=green&label=last-updated&logo=update&style=flat-squre) [![Contributor](https://img.shields.io/static/v1?label=by&message=Lujule&color=blue&style=flat-squre)](https://github.com/Lujule)
 

This technique reduces the adaptation difficulty often caused by poor performance on out-of-distribution test data before adaptation.

A curated list of temporal action localization/detection and related area (e.g. temporal action proposal) resources.

---

Contributors: Runhao Zeng, Qi Deng, Huixuan Xu, Shuaicheng Niu, Jian Chen

## <span id = "tal"> **Dataset** </span>
- [UCF101_noise](https://files.icg.tugraz.at/d/3551df694e3d4d6b89da/?p=%2Fucf_corrupted_videos&mode=list)
- [UCF101_contrast](https://drive.google.com/file/d/13QHfzMlu8Vjoo6NtmJGiI_whejyKTCKK/view?usp=drive_link)

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

