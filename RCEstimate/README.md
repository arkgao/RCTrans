# RCNet Training and Evaluation

This repository provides the training and evaluation framework for RCNet.

## Preparation

We release our synthetic dataset on [Hugging Face](https://huggingface.co/datasets/arkgao/NaturalRefractiveCorrespondence). Please follow its instructions to download and prepare the dataset.

## Usage

### 1. Training

Update `dataroot` in `configs/train.yaml` to your local dataset path, then run:
```bash
python train.py --opt configs/train.yaml
```

This will train RCNet and perform validation. Logs, checkpoints, and validation results are saved under `experiments/<experiment_name>/`, while TensorBoard summaries are written to `tb_logger/`.

We provide our trained model at `./pretrained_model/rcnet.pth` for reproduction purposes.

### 2. Validation / Testing

Update `dataroot` and `path.pretrain_network` in `configs/test_validation.yaml` to point to your dataset and checkpoint (or use our provided checkpoint), then run:
```bash
python test.py --opt configs/test_validation.yaml
```

This will evaluate the network and compute metrics. Results are saved in `results/<experiment_name>/`. You can set `list_file` to `basicshape_val_file.txt` or `omniobject_val_file.txt` to evaluate on specific validation subsets.

### 3. Reconstruction Testing

Update `path.pretrain_network` in `configs/test_recon.yaml` to your checkpoint (or use our provided checkpoint).

This script would save more detailed outputs and is used for network inference during 3D reconstruction. It is called automatically by the reconstruction code and does not need to be run manually.

## Dataset Details

We will release the dataset creation code along with detailed documentation.

## Acknowledgement

The project framework is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), and the network architecture implementation is based on [UniMatch](https://github.com/autonomousvision/unimatch). We thank the authors for their excellent work.

