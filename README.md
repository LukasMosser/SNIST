# Seismic-NIST
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


### Seismic-NIST - SNIST
| File            | Examples | Download (NumPy format)      |
|-----------|--------------|------------------------------|
| Training Waveforms | 600             | [train_amplitudes.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/train/train_amplitudes.npy) (13MB) |
| Training Velocities | 600             | [train_velocities.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/train/train_velocities.npy) (21KB) |
| Testing Amplitudes 0 (no noise - SNIST-0)  | 150             | [test_amplitudes.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_amplitudes.npy) (3MB) |
| Testing Amplitudes 1 (1 sigma noise - SNIST-1)  | 150             | [test_amplitudes.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_amplitudes.npy) (3MB) |
| Testing Amplitudes 2 (2 sigma noise - SNIST-2)  | 150             | [test_amplitudes.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_amplitudes.npy) (3MB) |
| Testing Velocities  | 150            | [test_velocities.npy](https://raw.githubusercontent.com/LukasMosser/SNIST/master/data/test/test_velocities.npy) (5KB)|

### Benchmarks and Results
|Model                            | SNIST-0 | SNIST-1 | SNIST-2 | Credit
|---------------------------------|---------|---------|---------|-------
|[1-Hidden Layer Benchmark](benchmarks/SNIST-Benchmark-Roeth-and-Tarantola.ipynb)     | 242.42 [m\s] | 287.98 [m\s] | 428.59% [m\s] | [@porestar](twitter.com/porestar)

### Generating the data
The data can be reproduced by running ```make build``` in the ```data_generation```
directory.  
This will run three scripts:
- ```generate_velocities.py```: creates the velocity models based on the paper by Roeth and Tarantola
- ```generate_amplitudes.sh```: runs a docker container of [devito](https://github.com/opesci/devito) and runs the forward model on the created velocities
- ```generate_noisy_test_set.py```: creates the noisy SNIST versions SNIST-1 and SNIST-2
