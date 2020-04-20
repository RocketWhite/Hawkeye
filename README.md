# Hawkeye

Hawkeye is a state-of-art adversarial examples detector(See more intro of adversarial examples [here](https://openai.com/blog/adversarial-example-research/)). (Paper Preparing)

## Getting Started
We highly recommend using anaconda to manage the virtual environment. You can download the anaconda [here](https://www.anaconda.com/distribution/). Please choose Python 3.7 version.

### Prerequisites

* python 3.7
* pytorch 1.4.0
* torchvision 0.5.0
* torchattacks 1.3

### Installing
```
git clone https://github.com/RocketWhite/Hawkeye.git

```
### Precaution
**WARNING** :: All images should be scaled to [0, 1] with transform[to.Tensor()] before used in attacks and detecting.

### Running
* In order to run the code, please set the right parameters in config.ini file.
* You can set which gpu you want to use in main.py

```
python main.py
```

### config.ini


## Implement your own experiment
We offer different componets for you to implement your own adversarial defense method. Check experiment/experiment_template.py and write your own experiment.
