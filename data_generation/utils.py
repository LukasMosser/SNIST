import numpy as np
import matplotlib.pyplot as plt
import torch
import random

def transform(y, nz):
    model_true = torch.ones(nz, 1)    
    for i, v in enumerate(y):
        model_true[40*i:40*(i+1), :] = v.item()
        
    return model_true

def plot_velocity_models(ax, models, colors=["blue", "black", "beige"], legend="Data", alpha=0.5):
    dev = models[:, :, 0].std(0).numpy()
    mean = models[:, :, 0].mean(0).numpy()
    sc = ax.plot(mean, color=colors[0], label=legend)
    upper = ax.plot(mean+dev, linestyle="--", color=colors[1])
    lower = ax.plot(mean-dev, linestyle="--", color=colors[1])
    sc2 = ax.fill_between(range(mean.shape[0]), mean-dev, mean+dev, alpha=alpha, color=colors[2])
    for curve in models[:, :, 0].numpy():
        ax.plot(range(len(curve)), curve, color=colors[0], alpha=0.01)


def plot_wiggle_traces(fig, xample, n_recorders):
    for i in range(1, n_recorders):
        ax = fig.add_subplot(n_recorders, 1, i)
        ax.plot(xample[:, -i], color="black")
        if i != n_recorders-1:
            ax.set_axis_off()
        else:
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.set_xlabel("Time [10 ms]", fontsize=20)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True
