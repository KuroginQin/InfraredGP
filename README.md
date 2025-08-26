# InfraredGP: Efficient Graph Partitioning via Spectral Graph Neural Networks with Negative Corrections

This repository provides a reference implementation of InfraredGP introduced in the paper "InfraredGP: Efficient Graph Partitioning via Spectral Graph Neural Networks with Negative Corrections", which won the Honorable Mention of [IEEE HPEC 2025 Graph Challenge](https://graphchallenge.mit.edu/champions).

### Abstract
Graph partitioning (GP), a.k.a. community detection, is a classic problem that divides nodes of a graph into densely-connected blocks. From a perspective of graph signal processing, we find that graph Laplacian with a negative correction can derive graph frequencies beyond the conventional range [0, 2]. To explore whether the low-frequency information beyond this range can encode more informative properties about community structures, we propose InfraredGP. It (i) adopts a spectral GNN as its backbone combined with low-pass filters and a negative correction mechanism, (ii) only feed random inputs to this backbone, (iii) derives graph embeddings via one feed-forward propagation (FFP) without any training, and (iv) derive feasible GP results by feeding the derived embeddings to BIRCH. Surprisingly, our experiments demonstrate that only based on the negative correction mechanism that amplifies low-frequency information beyond [0, 2], InfraredGP can derive distinguishable embeddings for some standard clustering modules (e.g., BIRCH) and obtain high-quality results for GP without any training. Following the IEEE HPEC Graph Challenge benchmark, we evaluate InfraredGP for both static and streaming GP, where InfraredGP can achieve much better efficiency (e.g., 16x-23x faster) and competitive quality over various baselines.

### Citing
If you find this project useful for your research, please cite the following paper.

```
TBD
```

If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
- numpy
- scipy
- pytorch
- scikit-learn
- munkres
- graph-tool

### Usage
The generated large datasets can be downloaded via this [link](https://drive.google.com/file/d/1Rv6kHvpoBQwql0rdn7IBeahbVXTRwqlK/view?usp=sharing). Please unzip the file and put datasets under ./data.

To run InfraredGP for static GP on dataset w/ a specific setting of N
```
python infraredgp_static_GP.py --N 5000 --tau -6 --L 10 --d 64
python infraredgp_static_GP.py --N 10000 --tau -3 --L 9 --d 64
python infraredgp_static_GP.py --N 50000 --tau -80 --L 40 --d 32
python infraredgp_static_GP.py --N 100000 --tau -80 --L 60 --d 32
python infraredgp_static_GP.py --N 500000 --tau -80 --L 70 --d 32
python infraredgp_static_GP.py --N 1000000 --tau -80 --L 60 --d 32
```
To run the standard static algorithm of InfraredGP (i.e., Algorithm 1) for streaming GP on a dataset w/ a specific setting of N & ind-th graph (ind=[0,1,2,3,4])
```
python infraredgp_stream_GP_base.py --N 100000 --ind 0 --tau -100 --L 20 --d 32
python infraredgp_stream_GP_base.py --N 1000000 --ind 0 --tau -100 --L 20 --d 16
```
To run the extended streaming algorithm of InfraredGP (i.e., Algorithm 2) for streaming GP on a dataset w/ a specific setting of N & ind-th graph (ind=[0,1,2,3,4])
```
python infraredgp_stream_GP.py --N 100000 --ind 0 --tau -100 --L 20 --d 32
python infraredgp_stream_GP.py --N 1000000 --ind 0 --tau -100 --L 20 --d 16
```

Please note that different environment setups (e.g., CPU, GPU, memory size, versions of libraries and packages, etc.) may result in different evaluation results regarding the inference time. When testing the inference time, please also make sure that there are no other processes with heavy resource requirements (e.g., GPUs and memory) running on the same server. Otherwise, the evaluated inference time may not be stable. The evaluation of GP quality may also be time-consuming in some cases.