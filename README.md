# matrix-multiplication

Resource usage of matrix multiplication - a comparison between CPU and GPU

## Requirements

Key dependencies

```
cupy
numpy
scipy
statsmodels
matplotlib
pandas
tqdm
psutil
gpuinfo
```

Install with conda on a linux-64 system

```
conda env create -f environment.yaml
conda activate matrix-multiplication
```

## Run

```
python multiply.py -p cpu/naive
python multiply.py -p cpu/row
python multiply.py -p gpu/naive
python multiply.py -p gpu/row
```

## Plot

```
python plot.py
```
