import os
import time
import psutil
import argparse
import cupy
from cupy.random import rand as gpu_matrix
import numpy
from numpy.random import rand as cpu_matrix
import pandas as pd
from tqdm import trange
from gpuinfo import GPUInfo
import multiprocessing as mp

# Matrix dimensions, A: m x n, B: n x m, and C: m x p
m = 1000000
n = 1000
p = 1


def multiplication_cpu():
    D = numpy.dot(cpu_matrix(m, n), numpy.dot(cpu_matrix(n, m), cpu_matrix(m, p)))
    numpy.save(file='data/cpu/naive/D.npy', arr=D)


def row_multiplication_cpu():
    # Perform multiplication starting with BC: n x p whose output requires less memory compared to AB: m x m
    BC = numpy.zeros((n, p))
    for i in trange(n):
        B_row = cpu_matrix(1, m)
        C = cpu_matrix(m, p)
        BC[i,:] = numpy.dot(B_row, C)

    # Perform final matrix multiplication ABC to retrieve D: m x p
    D = numpy.zeros((m, p))
    for i in trange(m):
        A_row = cpu_matrix(1, n)
        D[i,:] = numpy.dot(A_row, BC)

    cupy.save(file='data/cpu/row/D.npy', arr=D)


def multiplication_gpu():
    D = cupy.dot(gpu_matrix(m, n), cupy.dot(gpu_matrix(n, m), gpu_matrix(m, p)))
    cupy.save(file='data/gpu/naive/D.npy', arr=D)


def row_multiplication_gpu():
    # Perform multiplication starting with BC: n x p whose output requires less memory compared to AB: m x m
    BC = cupy.zeros((n, p))
    for i in trange(n):
        B_row = gpu_matrix(1, m)
        C = gpu_matrix(m, p)
        BC[i,:] = cupy.dot(B_row, C)

    # Perform final matrix multiplication ABC to retrieve D: m x p
    D = cupy.zeros((m, p))
    for i in trange(m):
        A_row = gpu_matrix(1, n)
        D[i,:] = cupy.dot(A_row, BC)

    cupy.save(file='data/gpu/row/D.npy', arr=D)


def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log resource usage of worker_process every 10 ms
    cpu_usage = []
    ram_usage = []
    gpu_usage = []
    gpu_memory = []
    while worker_process.is_alive():
        cpu_usage.append(p.cpu_percent())
        ram_usage.append(p.memory_info().rss)
        percent, memory = GPUInfo.gpu_usage()
        gpu_usage.append(percent[0])
        gpu_memory.append(memory[0])
        time.sleep(0.01)

    worker_process.join()
    return cpu_usage, ram_usage, gpu_usage, gpu_memory


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        '-p', '--path', required=True, type=str,
        choices=['cpu/naive', 'cpu/row', 'gpu/naive', 'gpu/row']
    )
    args = arg_parser.parse_args()

    try:
        os.makedirs(f'data/{args.path}')
    except FileExistsError:
        pass

    fn_map = dict(zip(
        ['cpu/naive', 'cpu/row', 'gpu/naive', 'gpu/row'],
        [multiplication_cpu, row_multiplication_cpu, multiplication_gpu, row_multiplication_gpu]
    ))

    cpu_usage, ram_usage, gpu_usage, gpu_memory = monitor(target=fn_map[args.path])
    resource_usage = pd.DataFrame({
        'cpu': cpu_usage,
        'ram': ram_usage,
        'gpu': gpu_usage,
        'gpu_mem': gpu_memory
    })
    resource_usage.to_csv(f'data/{args.path}/resource_usage.csv')
