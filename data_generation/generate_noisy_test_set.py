import numpy as np 
import torch
from utils import set_seed

def main():
    train_set = torch.from_numpy(np.load("../data/train/train_amplitudes.npy"))
    test_set = torch.from_numpy(np.load("../data/test/test_amplitudes.npy"))

    train_mean, train_std = train_set.mean(), train_set.std()

    sigma = 1.0
    set_seed(42)
    test_set_noise_1 = test_set.clone()
    test_set_noise_1 += torch.randn_like(test_set_noise_1)*train_std*sigma

    np.save("../data/test/test_amplitudes_noise_1.npy", test_set_noise_1.numpy())

    sigma = 2.0
    set_seed(42)
    test_set_noise_2 = test_set.clone()
    test_set_noise_2 += torch.randn_like(test_set_noise_2)*train_std*sigma

    np.save("../data/test/test_amplitudes_noise_2.npy", test_set_noise_2.numpy())

    return True


if __name__ == "__main__":
    main()
