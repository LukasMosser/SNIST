import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from generation import sample_model_N_times
from utils import plot_velocity_models

def main():
    dx = 5.0
    dz = 40
    nz = 360
    ny = 500

    v0 = 1500
    v_max = 4000
    dv_l_const = 190.
    n_layers = 9

    train_data_seed = 42
    test_data_seed = 7

    N_train = 600
    N_test = 150

    train_data_dir = '../data/train/'
    test_data_dir = '../data/test/'

    torch.manual_seed(train_data_seed)
    torch.cuda.manual_seed_all(train_data_seed)

    dVel0 = torch.distributions.Uniform(low=-150, high=150)
    dVel =  torch.distributions.Uniform(low=-380, high=380)

    train_models_th, train_labels_th = sample_model_N_times(dVel0, dVel, 
                                            v0, dv_l_const, v_max, 
                                            n_layers, dz, dx, ny, nz, 
                                            N_train)

    np.save(train_data_dir+'train_velocities.npy', train_labels_th.numpy())
    print("Succesfully generated training velocities.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plot_velocity_models(ax, train_models_th, colors=["black",  "black", "beige"], legend="Train")
    plt.legend(fontsize=24)
    fig.savefig("./figures/train_data.png", dpi=300, bbox_inches="tight")

    torch.manual_seed(test_data_seed)
    torch.cuda.manual_seed_all(test_data_seed)

    dVel0 = torch.distributions.Uniform(low=-150, high=150)
    dVel =  torch.distributions.Uniform(low=-380, high=380)

    test_models_th, test_labels_th = sample_model_N_times(dVel0, dVel, 
                                            v0, dv_l_const, v_max, 
                                            n_layers, dz, dx, ny, nz, 
                                            N_test)

    np.save(test_data_dir+'test_velocities.npy', test_labels_th.numpy())
    print("Succesfully generated test velocities.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plot_velocity_models(ax, test_models_th, colors=["red",  "red", "lightblue"], legend="Test")
    plt.legend(fontsize=24)
    fig.savefig("./figures/test_data.png", dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    main()