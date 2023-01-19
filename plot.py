import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

paths = ['cpu/row', 'cpu/naive', 'gpu/row', 'gpu/naive']


ecdf_dict = {}
resource_dict = {}
for path in paths:
    # Load data
    D = np.load(f'data/{path}/D.npy')
    # Empirical cumulative distribution function
    ecdf = ECDF(D.squeeze())
    # Store in dict
    ecdf_dict[path] = ecdf

    resource_usage = pd.read_csv(f'data/{path}/resource_usage.csv')
    resource_dict[path] = resource_usage


for path in paths:
    plt.plot(ecdf_dict[path].x, ecdf_dict[path].y, label=path)
plt.legend()
plt.xlabel('D_ij')
plt.ylabel('Cumulative probability')
plt.savefig('figs/ecdf.png')
plt.close()


for path in paths:
    time = resource_dict[path].index
    cpu_usage = resource_dict[path].cpu
    plt.plot(time, cpu_usage, label=path)
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('CPU usage (%)')
plt.savefig('figs/cpu_usage.png')
plt.close()


for path in paths:
    time = resource_dict[path].index
    ram_usage = resource_dict[path].ram
    plt.plot(time, ram_usage / 10e5, label=path)
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('RAM usage (MB)')
plt.savefig('figs/ram_usage.png')
plt.close()


for path in paths:
    time = resource_dict[path].index
    gpu_usage = resource_dict[path].gpu
    plt.plot(time, gpu_usage, label=path)
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('GPU usage (%)')
plt.savefig('figs/gpu_usage.png')
plt.close()


for path in paths:
    time = resource_dict[path].index
    gpu_mem_usage = resource_dict[path].gpu_mem
    plt.plot(time, gpu_mem_usage, label=path)
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('GPU memory usage (MB)')
plt.savefig('figs/gpu_mem_usage.png')
plt.close()
