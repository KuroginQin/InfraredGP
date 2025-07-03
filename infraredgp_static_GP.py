# InfraredGP for static GP

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
    parser.add_argument('--tau', type=int, default=-80)
    parser.add_argument('--L', type=int, default=60)
    parser.add_argument('--d', type=int, default=32)
    args = parser.parse_args()

    # ====================
    N = args.N # Number of nodes
    tau = args.tau # (Negative) correction term
    L = args.L # Number of GNN layers (i.e., iterations)
    emb_dim = args.d # Embedding dimensionality
    # ==========
    alpha = 1.0
    num_snap = 5 # Number of snapshots
    mu = 2.5 # ratio_within_over_between
    beta = 3.0 # block_size_heterogeneity

    # ====================
    pkl_file = open('data/static_%d_%d_%.1f_%.1f_edges_list.pickle' % (num_snap, N, mu, beta), 'rb')
    edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('data/static_%d_%d_%.1f_%.1f_gnd_list.pickle' % (num_snap, N, mu, beta), 'rb')
    gnd_list = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    time_list = []
    PCN_list = []
    RCL_list = []
    F1_list = []
    ARI_list = []
    # ==========
    for t in range(num_snap):
        # ====================
        print('SNAP-%d/%d' % (t+1, num_snap))
        edges = edges_list[t]
        gnd = gnd_list[t]
        # ==========
        num_nodes = len(gnd)
        num_edges = len(edges)
        num_clus = np.max(gnd) + 1

        # ====================
        degs = [0.0 for _ in range(num_nodes)]
        for (src, dst) in edges:
            # ==========
            degs[src] += 1.0
            degs[dst] += 1.0
        deg_min = np.min(degs)
        deg_max = np.max(degs)
        print('DEG MIN %d DEG MAX %d' % (deg_min, deg_max))

        # ====================
        ESP = 1e-3
        taus = []
        if tau < 0:
            tau_base = np.abs(tau)
            for i in range(num_nodes):
                taus.append(-min(tau_base, degs[i] - ESP))
        elif tau >= 0:
            taus= [tau for _ in range(num_nodes)]

        # ====================
        # Sparse regularized graph Laplacian
        idxs = []
        vals = []
        for (src, dst) in edges:
            # ==========
            v = alpha / (np.sqrt(degs[src] + taus[src]) * np.sqrt(degs[dst] + taus[dst]))
            # ==========
            idxs.append((src, dst))
            vals.append(v)
            # ==========
            idxs.append((dst, src))
            vals.append(v)
        for idx in range(num_nodes):
            idxs.append((idx, idx))
            vals.append(0.1) # theta
        # ==========
        idxs_tnr = torch.LongTensor(idxs).to(device)
        vals_tnr = torch.FloatTensor(vals).to(device)
        sup_sp = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                          torch.Size([num_nodes, num_nodes])).to(device)
        # ====================
        # Random noise input
        emb = get_rand_proj_mat(num_nodes, emb_dim, rand_seed=rand_seed_gbl)
        emb = torch.FloatTensor(emb).to(device)

        # ====================
        time_s = timeit.default_timer()
        for t in range(L):
            emb = torch.spmm(sup_sp, emb)
            emb = torch.tanh(emb)
            # ==========
            # Column-wise z-socre norm
            for j in range(emb_dim):
                crt_mean = torch.mean(emb[:, j])
                crt_std = torch.std(emb[:, j])
                emb[:, j] = (emb[:, j] - crt_mean) / crt_std
        # ==========
        emb = torch.sigmoid(emb)
        # ==========
        time_e = timeit.default_timer()
        emb_time = time_e - time_s

        # ====================
        if torch.cuda.is_available():
            emb = emb.cpu().data.numpy()
        else:
            emb = emb.data.numpy()

        # ====================
        time_s = timeit.default_timer()
        clus = Birch(n_clusters=None, threshold=0.6, branching_factor=50)
        clus_res = clus.fit_predict(emb)
        time_e = timeit.default_timer()
        clus_time = time_e - time_s

        # ====================
        run_time = emb_time + clus_time
        print('TIME %f (EMB %f CLUS %f)' % (run_time, emb_time, clus_time))

        # ====================
        ARI, RCL, PCN = quality_eva(gnd, clus_res)
        F1 = 2.0*PCN*RCL / (PCN+RCL)
        print('F1 %.4f (PCN %.4f RCL %.4f) ARI %.4f' %
              (F1, PCN, RCL, ARI))
        print()
        # ==========
        time_list.append(run_time)
        PCN_list.append(PCN)
        RCL_list.append(RCL)
        F1_list.append(F1)
        ARI_list.append(ARI)

    # =====================
    time_mean = np.mean(time_list)
    time_std = np.std(time_list)
    PCN_mean = np.mean(PCN_list)
    PCN_std = np.std(PCN_list)
    RCL_mean = np.mean(RCL_list)
    RCL_std = np.std(RCL_list)
    F1_mean = np.mean(F1_list)
    F1_std = np.std(F1_list)
    ARI_mean = np.mean(ARI_list)
    ARI_std = np.std(ARI_list)
    print('InfraredGP - Static N=%d' % (N))
    print('TIME %.4f~(%.4f) F1 %.4f~(%.4f) '
          'PCN %.4f~(%.4f) RCL %.4f~(%.4f) ARI %.4f~(%.4f)' %
          (time_mean, time_std, F1_mean, F1_std, PCN_mean, PCN_std, RCL_mean, RCL_std,
           ARI_mean, ARI_std,))
