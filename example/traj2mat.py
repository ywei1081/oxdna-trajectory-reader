#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import h5py  # HDF5 for Python
from oxdna_trajectory_reader import Trajectory


def write_mat7_3_header(fn):
    with open(fn, 'r+b') as f:
        f.write(f'MATLAB 7.3 MAT-file, Platform: PCWIN64, Created on: {time.asctime()} HDF5 schema 1.00 .'.encode())
        f.write(b' ' * (116 - f.tell()))
        f.write(bytes.fromhex('0000 0000 0000 0000 0002 494d'))


def save_mat7_3(out_fn, variables, confs):
    print(f'saving to "{out_fn}"...', flush=True, end='')
    start_ts = time.time()
    with h5py.File(out_fn, mode='w', userblock_size=512) as f:
        for k, v in variables.items():
            f.create_dataset(k, data=v)
            f[k].attrs['MATLAB_class'] = np.bytes_(b'double')
        f.create_dataset('all_conf', shape=(len(confs), *confs[0].shape), dtype=np.float64)
        for i, conf in enumerate(confs):
            f['all_conf'][i] = conf
    write_mat7_3_header(out_fn)
    print(f'{time.time() - start_ts:.1f}s', flush=True)


def convert(traj_fn, out_fn):
    print(f'reading trajectory from "{traj_fn}"...', flush=True, end='')
    start_ts = time.time()
    traj = Trajectory(traj_fn, chunk_size=1000)
    times = []
    boxes = []
    energies = []
    confs = []
    for conf in traj:
        times.append(conf.time)
        boxes.append(conf.box)
        energies.append(conf.energy)
        confs.append(conf._nucleotides)
    variables = {
        'all_time': np.array(times, dtype=np.float64),
        'all_box': np.array(boxes, dtype=np.float64),
        'all_energy': np.array(energies, dtype=np.float64),
    }
    print(f'{time.time() - start_ts:.1f}s', flush=True)
    save_mat7_3(out_fn, variables, confs)


def convert_iter(traj_fn, out_fn):
    print(f'converting trajectory from "{traj_fn}"...', flush=True)
    start_ts = time.time()

    traj = Trajectory(traj_fn)
    conf = traj[0]
    shapes = {
        'all_time': (traj.length,),
        'all_box': (traj.length, *conf.box.shape),
        'all_energy': (traj.length, *conf.energy.shape),
        'all_conf': (traj.length, *conf._nucleotides.shape),
    }
    with h5py.File(out_fn, mode='w', userblock_size=512) as f:
        for k, v in shapes.items():
            f.create_dataset(k, shape=v, dtype=np.float64)
            f[k].attrs['MATLAB_class'] = np.bytes_(b'double')

        for i, conf in enumerate(traj):
            f['all_time'][i] = conf.time
            f['all_box'][i] = conf.box
            f['all_energy'][i] = conf.energy
            f['all_conf'][i] = conf._nucleotides
            if i % 300 == 0 and i > 0:
                print(f'{i / traj.length * 100:.1f}% converted...', flush=True)

    write_mat7_3_header(out_fn)
    print(f'convertion done...{time.time() - start_ts:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Parse and save oxDNA trajectory data into MATLAB v7.3 MAT-file')
    parser.add_argument('trajectory')
    parser.add_argument('--outname')
    parser.add_argument('--iter', action='store_true', help='convert while reading, use less RAM')
    args = parser.parse_args()
    outname = args.outname or os.path.splitext(args.trajectory)[0] + '.mat'
    if not outname.endswith('.mat'):
        outname += '.mat'
    if args.iter:
        convert_iter(args.trajectory, outname)
    else:
        convert(args.trajectory, outname)


if __name__ == "__main__":
    main()
