# Standard static GP algorithm of InfraredGP for streaming GP

import torch
from sklearn.cluster import Birch

import argparse
import pickle
import random
import timeit
from utils import *

rand_seed_gbl = 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # ====================
    setup_seed(rand_seed_gbl)

    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=100000)
    parser.add_argument('--ind', type=int, default=0)
    parser.add_argument('--tau', type=int, default=-100)
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--d', type=int, default=32)
    args = parser.parse_args()

    # ====================
    N = args.N # Number of nodes
    snap_idx = args.ind # Snapshot index
    tau = args.tau # (Negative) correction term
    L = args.L # Number of GNN layers (i.e., iterations)
    emb_dim = args.d # Embedding dimensionality
    # ==========
    alpha = 1.0
    num_snap = 5 # Number of snapshots
    mu = 2.5 # ratio_within_over_between
    beta = 3.0 # block_size_heterogeneity
    # ==========
    num_step = 10  # Number of streaming steps
    num_nodes_per_step = int(N / num_step)

    # ====================
    pkl_file = open('data/stream_%d_%d_%.1f_%.1f_edges_list.pickle' % (snap_idx, N, mu, beta), 'rb')
    edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('data/stream_%d_%d_%.1f_%.1f_gnd.pickle' % (snap_idx, N, mu, beta), 'rb')
    gnd = pickle.load(pkl_file)
    pkl_file.close()

    # ===================
    node_idx_s = 0
    node_idx_e = num_nodes_per_step
    acc_edges = [] # Cumulative edge list
    for s in range(num_step):
        # ====================
        print('STEP-%d/%d' % (s+1, num_step))
        edges = edges_list[s]
        crt_edges = [(src-1, dst-1) for (src, dst) in edges]
        for (src, dst) in crt_edges:
            # ==========
            if src == dst: continue
            # ==========
            if src > dst:
                tmp = src
                src = dst
                dst = tmp
            acc_edges.append((src, dst))
        # ==========
        crt_gnd = gnd[0:node_idx_e] # Ground-truth w.r.t. current (cumulative) topo
        acc_num_nodes = node_idx_e # Cumulative number of nodes
        acc_num_edges = len(acc_edges) # Cumulative number of edges

        # ====================
        acc_degs = [0 for _ in range(acc_num_nodes)] # Cumulative node degree list
        acc_src_idxs = []
        acc_dst_idxs = []
        for (src, dst) in acc_edges:
            # ==========
            acc_degs[src] += 1
            acc_degs[dst] += 1
            # ==========
            acc_src_idxs.append(src)
            acc_dst_idxs.append(dst)

        # ====================
        ESP = 1e-3
        taus = []
        if tau < 0:
            tau_base = np.abs(tau)
            for i in range(acc_num_nodes):
                taus.append(-min(tau_base, acc_degs[i] - ESP))
        elif tau >= 0:
            taus = [tau for _ in range(acc_num_nodes)]

        # ====================
        idxs = []
        vals = []
        for (src, dst) in acc_edges:
            # ==========
            v = alpha / (np.sqrt(acc_degs[src] + taus[src]) * np.sqrt(acc_degs[dst] + taus[dst]))
            # ==========
            idxs.append((src, dst))
            vals.append(v)
            # ==========
            idxs.append((dst, src))
            vals.append(v)
        for idx in range(acc_num_nodes):
            idxs.append((idx, idx))
            vals.append(0.1) # theta
        # ==========
        idxs_tnr = torch.LongTensor(idxs).to(device)
        vals_tnr = torch.FloatTensor(vals).to(device)
        acc_sup_sp = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                              torch.Size([acc_num_nodes, acc_num_nodes])).to(device)
        # ====================
        # Random noise input
        crt_emb = get_rand_proj_mat(acc_num_nodes, emb_dim, rand_seed=rand_seed_gbl)
        crt_emb = torch.FloatTensor(crt_emb).to(device)

        # ====================
        time_s = timeit.default_timer()
        for t in range(L):
            crt_emb = torch.spmm(acc_sup_sp, crt_emb)
            crt_emb = torch.tanh(crt_emb)
            # ==========
            # Column-wise z-socre norm
            for j in range(emb_dim):
                z_mean = torch.mean(crt_emb[:, j])
                z_std = torch.std(crt_emb[:, j])
                crt_emb[:, j] = (crt_emb[:, j] - z_mean) / z_std
        # ==========
        crt_emb = torch.sigmoid(crt_emb)
        time_e = timeit.default_timer()
        emb_time = time_e - time_s

        # ====================
        if torch.cuda.is_available():
            crt_emb = crt_emb.cpu().data.numpy()
        else:
            crt_emb = crt_emb.data.numpy()

        # ====================
        # Run standard BRICH from scratch
        time_s = timeit.default_timer()
        clus = Birch(n_clusters=None, threshold=0.6, branching_factor=50)
        crt_clus_res = clus.fit_predict(crt_emb)
        time_e = timeit.default_timer()
        clus_time = time_e - time_s

        # ====================
        run_time = emb_time + clus_time
        print('TIME %f (EMB %f CLUS %f)' % (run_time, emb_time, clus_time))

        # ====================
        ARI, RCL, PCN = quality_eva(crt_gnd, crt_clus_res)
        F1 = 2.0*PCN*RCL / (PCN+RCL)
        print('F1 %.4f (PCN %.4f RCL %.4f) ARI %.4f ' %
              (F1, PCN, RCL, ARI))
        print()

        node_idx_s = node_idx_e
        node_idx_e = node_idx_s + num_nodes_per_step
