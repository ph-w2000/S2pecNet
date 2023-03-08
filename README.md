# S2pecNet

This repository provides the overall framework for training and evaluating audio anti-spoofing systems proposed in ['Robust Audio Anti-Spoofing with Fusion-Reconstruction Learning on Multi-Order Spectrograms'](https://arxiv.org/abs/2110.01200)

### Getting started
`requirements.txt` must be installed for execution. We state our experiment environment for those who prefer to simulate as similar as possible. 
- Installing dependencies
```
pip install -r requirements.txt
```
- Our environment (for GPU training)
  - GPU: 1 NVIDIA RTX A6000
    - About 45GB is required to train AASIST using a batch size of 48
  - gpu-driver: 470.63.01

### Data preparation
We train/validate/evaluate S2pecNet using the ASVspoof 2019 logical access dataset [4].
```
python ./download_dataset.py
```
(Alternative) Manual preparation is available via 
- ASVspoof2019 dataset: https://datashare.ed.ac.uk/handle/10283/3336
  1. Download `LA.zip` and unzip it
  2. Set your dataset directory in the configuration file

### Training 
The `main.py` includes train/validation/evaluation.

To train S2pecNet [1]:
```
python main.py --config ./config/S2pecNet.conf
```


### Pre-trained models
We provide pre-trained AASIST and AASIST-L.

To evaluate S2pecNet [1]:
- It shows `EER: 0.77%`, `min t-DCF: 0.0242`
```
python main.py --eval --config ./config/AASIST.conf
```

### Developing custom models
Simply by adding a configuration file and a model architecture, one can train and evaluate their models.

To train a custom model:
```
1. Define your model
  - The model should be a class named "Model"
2. Make a configuration by modifying "model_config"
  - architecture: filename of your model.
  - hyper-parameters to be tuned can be also passed using variables in "model_config"
3. run python main.py --config {CUSTOM_CONFIG_NAME}
```

### License
```

```

### Acknowledgements
This repository is built on top of several open source projects. 
- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
- [min t-DCF implementation](https://www.asvspoof.org/resources/tDCF_python_v2.zip)

The dataset we use is ASVspoof 2019 [4]
- https://www.asvspoof.org/index2019.html

### References
[1] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}
```

```
