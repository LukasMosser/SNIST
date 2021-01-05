[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/LukasMosser/SNIST.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/LukasMosser/SNIST/context:python) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/SNIST/blob/master/benchmarks/SNIST_Benchmark_Roeth_and_Tarantola.ipynb)

# Seismic-NIST
<details><summary>Table of Contents</summary><p>

* [What the data looks like](#what-the-data-looks-like)
* [Why Seismic-NIST was created](#why-seismic-nist-was-created)
* [Get the Data](#get-the-data)
* [Benchmarks and Results](#benchmarks-and-results)
* [Generating the data](#generating-the-data)
* [Contributing](#contributing)
* [License](#license)
</p></details><p></p>

```Seismic-NIST``` is a dataset of acoustic [seismic](https://wiki.seg.org/wiki/Seismic_Data_Analysis) waveforms and their underlying [velocity profiles](https://wiki.seg.org/wiki/Inversion_of_seismic_data). The dataset is inspired by the work of [Roeth and Tarantola 1994](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/93JB01563) where the authors tried to perform [seismic inversion](https://wiki.seg.org/wiki/Inversion_of_seismic_data) from raw acoustic waveforms at various levels of noise. Here we provide a reference dataset of such waveforms. The machine learning task to be solved is a regression problem of predicting synthetic [p-Wave velocity](https://wiki.seg.org/wiki/Dictionary:P-wave) profiles from given acoustic waveforms. The data can be generated completly from scratch using [```torch```](pytorch.org) and libraries from the [```devito```](https://github.com/opesci/devito) project.  
The dataset is named after the outstanding deep-learning benchmark [MNIST](http://yann.lecun.com/exdb/mnist/index.html) by [Yann Le Cun](http://yann.lecun.com/), and is inspired by other projects such as [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) and [KMNIST](https://github.com/rois-codh/kmnist).

## What the data looks like

The dataset consists of 750 waveforms generated from 9-layer earth models of acoustic p-wave velocities.  
The training set consists of 600 waveforms and the test sets consist of 150 waveforms.  
There are three test sets - SNIST-0, SNIST-1 and SNIST-2.  
The number corresponds to the level of noise added to the test set i.e. SNIST-0 has no noise added, SNIST-1 adds 1 sigma of noise, and SNIST-2 has 2 sigma of noise added. The noise is Gaussian uncorrelated noise.  
Each waveform consists of 20 [traces](https://wiki.seg.org/wiki/Dictionary:Seismic_trace) according to 20 [offsets](https://wiki.seg.org/wiki/Dictionary:Common-offset_gather) sampled at 8 ms time intervals. [p-Wave velocities](https://wiki.seg.org/wiki/Dictionary:P-wave) are capped at 4000 [m/s]. 
Here's what the waveform amplitudes and some of the velocity profiles (ground truth - black) look like.

![](benchmarks/figures/test_amplitudes_grid.png)
![](benchmarks/figures/test_velocities_grid.png)

## Why Seismic-NIST was created 

The dataset was largely inspired by discussion on the [software-underground](https://softwareunderground.org/) slack channel and by Agile Geoscience's [blog post](https://agilescientific.com/blog/2019/4/3/what-makes-a-good-benchmark-dataset) on benchmark studies in the machine learning - geoscience domain.    
While the realism and usefulness in terms of real seismic applications is limited, this benchmark may serve as a reference on what a realistic benchmark should include. Hence, this benchmark is very much a platform or sandbox as not (m)any reference benchmarks exist in the seismic deep-learning domain. It is up to the community to shape what we want out of such a reference benchmark and I hope to provide here a starting point for such a discussion.  
If you would like to contribute or would like to raise an [issue](https://github.com/LukasMosser/SNIST/issues) please do so and join the discussion on the [slack-channel](https://softwareunderground.org/).

## Get the Data

The data comes prepackaged as ```.npy``` files. Which you can either download manually or use the existing ```torch.dataset``` implementation found in ```utils/snist```.

| File            | Examples | Size | Download (NumPy format)      |
|-----------|--------------|------------|------------------|
| Training Amplitudes | 600             | 13 MB | [train_amplitudes.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/train/train_amplitudes.npy) |
| Training Velocities | 600             | 21 KB | [train_velocities.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/train/train_velocities.npy)  |
| Testing Amplitudes 0 (no noise - SNIST-0)  | 150             | 3 MB |[test_amplitudes.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_amplitudes.npy)  |
| Testing Amplitudes 1 (1 sigma noise - SNIST-1)  | 150             | 3 MB |[test_amplitudes_noise_1.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_amplitudes_noise_1.npy)  |
| Testing Amplitudes 2 (2 sigma noise - SNIST-2)  | 150             | 3 MB |[test_amplitudes_noise_2.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_amplitudes_noise_2.npy)  |
| Testing Velocities  | 150            | 5 KB | [test_velocities.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_velocities.npy) |

## Pytorch Datasets

The following is an example on how to use the provided dataset in ```torch```.
All the data will automatically be downloaded - in this case - from the directory and is ready for training.  
You can try it out on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/SNIST/blob/master/benchmarks/SNIST_Benchmark_Roeth_and_Tarantola.ipynb).

```python
from snist.dataset import SNIST

snist_train = SNIST('./', train=True, download=True)
snist_0_test = SNIST('./', train=False, download=True, noise=0)
snist_1_test = SNIST('./', train=False, download=True, noise=1)
snist_2_test = SNIST('./', train=False, download=True, noise=2)
```

## Benchmarks and Results

A reference implementation [is provided](benchmarks/SNIST_Benchmark_Roeth_and_Tarantola.ipynb) and here we collect the performance of methods that have been evaluated on the SeismicNIST dataset.  
If you wish to contribute to this list please raise a [pull-request](https://github.com/LukasMosser/SNIST/pulls) and provide a link to a repository where your results can be reproduced.    

|Model                            | SNIST-0 | SNIST-1 | SNIST-2 | Credit | Link 
|---------------------------------|---------|---------|---------|--------|------
|[1-Hidden Layer Benchmark](benchmarks/SNIST-Benchmark-Roeth-and-Tarantola.ipynb)     | 242.42 [m\s] | 287.98 [m\s] | 428.59 [m\s] | [@porestar](twitter.com/porestar)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/SNIST/blob/master/benchmarks/SNIST_Benchmark_Roeth_and_Tarantola.ipynb)

## Generating the data

The data can be reproduced by running ```make build``` in the ```data_generation```
directory.  
This will run three scripts:
- ```generate_velocities.py```: creates the velocity models based on the paper by Roeth and Tarantola
- ```generate_amplitudes.sh```: runs a docker container of [devito](https://github.com/opesci/devito) and runs the forward model on the created velocities
- ```generate_noisy_test_set.py```: creates the noisy SNIST versions SNIST-1 and SNIST-2

## Contributing

If you would like to contribute or would like to raise an [issue](https://github.com/LukasMosser/SNIST/issues) please do so and join the discussion on the [slack-channel](https://softwareunderground.org/).

## License

MIT License

Copyright (c) 2019 Lukas Mosser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
