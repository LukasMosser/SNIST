import numpy as np
import os

os.environ['DEVITO_OPENMP'] = '1'

from argparse import ArgumentParser

from devito.logger import info
from devito import Constant, Function, smooth
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry, Model
import os
import sys

def to_model(velocities, N_x=500):
    velo_model = []
    for vel in velocities:
        velo_model.append([vel]*40)
    return np.repeat(np.array(velo_model).reshape(len(velocities)*40, 1), N_x, axis=1)

def parse_args(argv):
    """Parse Commandline Arguments
    
    Arguments:
        argv {dict} -- Contains command line arguments
    
    Returns:
        ArgparseArguments -- Parsed commandline arguments
    """

    description = ("Wave-solver for Roeth and Tarantola 1994 Example")
    parser = ArgumentParser(description=description)
    parser.add_argument('-nd', dest='ndim', default=2, type=int,
                        help="Preset to determine the number of dimensions")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=0,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("--input", type=str, help="Name of input velocity model file.")
    parser.add_argument("--output", type=str, help="Name of output waveform model file.")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")

    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")
    parser.add_argument("--checkpointing", default=False, action='store_true',
                        help="Use Checkpointing?")
    args = parser.parse_args(argv)

    return args 

if __name__ == "__main__":
    #parse parameters
    args = parse_args(sys.argv[1:])

    #Simulation Parameters
    tn = 2710.0
    num_samples = 271

    #Geometry Parameters
    shape = tuple([2000, 360])#tuple([500, 360])
    spacing = tuple([5.0, 5.0])
    dist_first_receiver = 140. #meters
    spacing_receivers = 90. #meters
    num_receivers = 21

    #Wavelet Parameters
    peak_frequency = 8. #Hz

    vp = np.load(os.path.expandvars(args.input)).astype(np.float32)/1000. #convert from m/s to km/s
    print(vp.shape)
    amps = []

    #Define the basic Model container
    origin = tuple([0. for _ in shape])
    dtype = np.float32
    
    kernel='OT2'
    space_order=6
    m_i = to_model(np.array([4.0]*9), N_x=shape[0]).T
    model = Model(space_order=space_order, vp=m_i, origin=origin, shape=shape, dtype=dtype, spacing=spacing, nbpml=0)
    
    # Source and receiver geometries
    src_coordinates = np.zeros((1, len(spacing)))    
    src_coordinates[0, :] = 0.
    src_coordinates[:, 0] = spacing[0]*500.
    rec_coordinates = np.zeros((num_receivers, len(spacing)))
    rec_coordinates[:, 0] = 500.*spacing[0]+dist_first_receiver+spacing_receivers*np.array(range(num_receivers))
    rec_coordinates[0, :] = 0.
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0=0.0, tn=tn, src_type='Ricker', f0=peak_frequency/1000.)
    
    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, geometry, kernel=kernel, space_order=space_order, autotune=args.autotune, dse=args.dse, dle=args.dle)

    save = False
    autotune = True 

    print("Generating Amplitudes")
    for i, v in enumerate(vp):
        m = to_model(v, N_x=shape[0]).T
        #Define the basic Model container
        solver.model.vp = m
        rec, u, summary = solver.forward(save=save, autotune=args.autotune)
        #Resample to 'num_samples' samples and store
        amps.append(rec.resample(num=num_samples).data)
        if i % 10 == 9:
            print("Generated Model ", str(i))
    np.save(os.path.expandvars(args.output), np.array(amps)[:, :, 1:])
