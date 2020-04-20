# Hawkeye

Hawkeye is a state-of-art adversarial examples detector(See more intro of adversarial examples [here](https://openai.com/blog/adversarial-example-research/)). (Paper Preparing)

## Getting Started
We highly recommend using anaconda to manage the virtual environment. You can download the anaconda [here](https://www.anaconda.com/distribution/). Please choose Python 3.7 version.

### Prerequisites

* python 3.7
* pytorch 1.4.0
* torchvision 0.5.0
* torchattacks 1.1

### Installing
```
git clone https://github.com/RocketWhite/Hawkeye.git

```
### Precaution
* **WARNING** :: All images should be scaled to [0, 1] with transform[to.Tensor()] before used in attacks and detecting.

## Structure
```
├───Hawkeye
│   │   config.ini                          # configuration file.
│   │   environment.yaml                    # conda environment yaml
│   │   main.py                             # main detect process
│   │
│   ├───attacks
│   │       attack_wrapper.py
│   │       __init__.py
│   │
│   ├───classifiers
│   │       nnclassifier.py
│   │       __init__.py
│   │
│   ├───datasets
│   │   └───MNIST
│   │       ├───processed
│   │       │       test.pt
│   │       │       training.pt
│   │       │
│   │       └───raw
│   │               t10k-images-idx3-ubyte
│   │               t10k-images-idx3-ubyte.gz
│   │               t10k-labels-idx1-ubyte
│   │               t10k-labels-idx1-ubyte.gz
│   │               train-images-idx3-ubyte
│   │               train-images-idx3-ubyte.gz
│   │               train-labels-idx1-ubyte
│   │               train-labels-idx1-ubyte.gz
│   │
│   ├───detectors
│   │   │   hawkeye.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           hawkeye.cpython-37.pyc
│   │           __init__.cpython-37.pyc
│   │
│   ├───models
│   │   │   renset.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           renset.cpython-37.pyc
│   │           __init__.cpython-37.pyc
│   │
│   ├───models_dict
│   │       ResNetCIFAR10.ckpt
│   │       ResNetMNIST.ckpt
│   │
│   ├───squeezers
│   │   │   colordepthsqueezer.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           colordepthsqueezer.cpython-37.pyc
│   │           __init__.cpython-37.pyc
│   │
│   └───utils
│           __init__.py
│
```
