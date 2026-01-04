# RCTrans: Transparent Object Reconstruction in Natural Scene via Refractive Correspondence Estimation
This is the official implementation of the paper "RCTrans: Transparent Object Reconstruction in Natural Scene via Refractive Correspondence Estimation". [SIGGRAPH Asia 2025]

![](asset/teaser.jpg)

## [Project page](https://arkgao.github.io/RCTrans/) | [Paper](docs/pdf/main.pdf)

# TODO
We will release all the codes to facilitate the community. For now, we have organized the code for the object reconstruction. But the codes for RCNet training/testing and codes for creating the training datasets involve complex frameworks. We are still organizing them to facilitate the reproduction with the least amount of configuration. We would try to release them all in Jan. 2026.

- [x] release reconstruction code
- [ ] release network training and testing code
- [ ] release dataset creation code
- [ ] release real data process code and guidance
- [x] release reconstruction data
- [ ] release training and validation dataset

# Setup
1. Install torch and torchvision according to your environment. For reference, we use torch==2.0.1+cu118.
2. Install other packages.
    ```shell
    git clone https://github.com/arkgao/RCTrans.git
    cd RCTrans
    pip install -r requirements.txt
    ```
# Object Reconstruction
All code related to object reconstruction is placed in the **TransRecon** directory.
It would leverage the pretrained RCNet to reconstruct transparent objects from multi-view images.

Please first switch to this directory 
``` shell
cd TransRecon
```
and then follow the instructions provided in [./TransRecon/README.md](./TransRecon/README.md) to run the code.

# RCNet Training & Testing

# Real Data Processing

# Citation
```
@inproceedings{10.1145/3757377.3763859,
author = {Gao, Fangzhou and Kang, Yuzhen and Zhang, Lianghao and Wang, Li and Wang, Qishen and Zhang, Jiawan},
title = {RCTrans: Transparent Object Reconstruction in Natural Scene via Refractive Correspondence Estimation},
year = {2025},
url = {https://doi.org/10.1145/3757377.3763859},
doi = {10.1145/3757377.3763859},
booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
articleno = {1},
numpages = {11},
}
```