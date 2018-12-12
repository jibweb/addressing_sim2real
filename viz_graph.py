from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from dataset import get_dataset, DATASETS
from utils.params import params as p
from utils.logger import log
from preprocessing import get_graph_preprocessing_fn


def viz_n_hop(adj, hop_nb):
    mt = adj
    for idx in range(hop_nb):
        mt = (np.matmul(adj, mt) > 0).astype(adj.dtype)

    plt.imshow(mt)
    plt.show()


def viz_lrf_coords(edge_feats):
    coords_lrf = edge_feats[:, :, 1:4]
    for i in range(p.nodes_nb):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        ax.scatter3D(coords_lrf[i, :, 0],
                     coords_lrf[i, :, 1],
                     coords_lrf[i, :, 2])
        plt.show()


def viz_pc(feats):
    assert p.feat_nb == 3
    xdata = feats[:, 0]
    ydata = feats[:, 1]
    zdata = feats[:, 2]
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.scatter3D(xdata, ydata, zdata)
    plt.show()


def check_for_nan(fn):
    try:
        feats, bias, edge_feats, valid_indices = feat_compute(fn)
    except:
        print fn
        return

    if np.mean(np.isnan(feats)) != 0. or \
            np.mean(np.isnan(bias)) != 0. or \
            np.mean(np.isnan(edge_feats)) != 0. or \
            np.mean(np.isnan(valid_indices)) != 0.:
        print fn[32:]
        print "% of feats NaN:", np.mean(np.isnan(feats))
        print "% of bias NaN:", np.mean(np.isnan(bias))
        print "% of edge_feats NaN:", np.mean(np.isnan(edge_feats))
        print "% of valid_indices NaN:", np.mean(np.isnan(valid_indices))

    return 1


if __name__ == "__main__":
    log("START\n")

    DATASET = DATASETS.ModelNet40
    Dataset, CLASS_DICT = get_dataset(DATASET)

    p.gridsize = 64
    p.nodes_nb = 16
    p.neigh_size = 0.401
    p.mesh = False
    p.edge_feats = True
    p.edge_feat_nb = 3
    p.feats_3d = True
    p.scale = True
    p.min_angle_z_normal = 10

    if p.mesh:
        p.neigh_nb = 1000
        p.feat_nb = 1000
    else:
        p.neigh_nb = 4
        if p.feats_3d:
            p.feat_nb = 800
        else:
            p.feat_nb = 3

    p.debug = True
    p.viz = True
    p.viz_small_spheres = True

    p.to_remove = 0.
    p.to_keep = 10000
    p.rotation_deg = 0

    feat_compute = get_graph_preprocessing_fn(p)
    regex = "/*_full_wnormals_wattention.ply" if p.mesh  \
            else "/*_full_wnormals_wattention.pcd"
    # regex = "/*_full_wnormals_wattention_TMP-RM.pcd"
    d = Dataset(batch_size=64, val_set_pct=0, regex=regex)

    datasets = d.get_train_test_dataset()
    if len(datasets) == 2:
        train_set, test_set = datasets
    elif len(datasets) == 3:
        train_set, test_set, val_set = datasets

    train_set = shuffle([item for sublist in test_set for item in sublist])

    for idx, fn in enumerate(train_set):
        log("{}/{} {}", idx+1, len(train_set), fn.split('/')[-1])
        feats, bias, edge_feats, valid_indices = feat_compute(fn)

        if np.sum(valid_indices) == 0:
            print "No valid indices found"
        if np.mean(np.isnan(feats)) != 0. or \
                np.mean(np.isnan(bias)) != 0. or \
                np.mean(np.isnan(edge_feats)) != 0. or \
                np.mean(np.isnan(valid_indices)) != 0.:
            print fn[32:]
            print "% of feats NaN:", np.mean(np.isnan(feats))
            print "% of bias NaN:", np.mean(np.isnan(bias))
            print "% of edge_feats NaN:", np.mean(np.isnan(edge_feats))
            print "% of valid_indices NaN:", np.mean(np.isnan(valid_indices))
            break

        # viz_pc(feats)
        # plt.imshow(bias)
        # plt.show()
