from colorsys import hsv_to_rgb
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.utils import shuffle
from time import sleep

from dataset import get_dataset, DATASETS
from utils.params import params as p
from utils.logger import log, logw
from preprocessing import get_graph_preprocessing_fn, PyGraph, NanNormals


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
    assert p.feat_nb >= 3
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


def viz_pcs(feats):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    for feat_idx, feat in enumerate(feats):
        ax = fig.add_subplot(4, 8, feat_idx+1, projection='3d')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        xdata = feat[:, 0]
        ydata = feat[:, 1]
        zdata = feat[:, 2]
        ax.scatter3D(xdata, ydata, zdata)

    plt.show()


def viz_tconv_indices(tconv_idx, sampled_nodes):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    node_colors = [hsv_to_rgb(float(i)/sampled_nodes, 1, 1)
                   for i in range(sampled_nodes)] + [0., 0., 0.]

    for node_idx, indices in enumerate(tconv_idx):
        ax = fig.add_subplot(4, 8, node_idx+1)
        img = np.zeros((3, 3, 3))
        for u in range(3):
            for v in range(3):
                img[u, v, :] = node_colors[indices[3*u + v]]
        ax.imshow(img)

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


def precompute_wfn(in_fn):
    xn = feat_compute(in_fn)
    return xn, in_fn


if __name__ == "__main__":
    log("START\n")

    p.define_from_file("params/ScanNet_COORDS_SET_GAT_MaxPool.yaml")

    # p.dataset = DATASETS.ModelNet40PLY
    p.dataset = DATASETS.ScanNetToModelNet40
    Dataset, CLASS_DICT = get_dataset(p.dataset)

    pregex = "/*_visible_normals_bin.ply"
    if p.dataset == DATASETS.ModelNet10.name:
        pregex = "/*_full_wnormals_wattention.ply"
    elif p.dataset == DATASETS.ModelNet10PLY.name:
        pregex = "/*[0-9]_visible_normals_bin.ply"  # Manifold models
        # pregex = "/*_bin_visible_normals_bin.ply"  # TSDF models
    elif p.dataset == DATASETS.ScanNet.name:
        pregex = "*_wnormals.ply"

    feat_compute = get_graph_preprocessing_fn(p)
    dataset = Dataset(batch_size=p.batch_size,
                      balance_train_set=False,
                      val_set_pct=0.0,  # )
                      regex=pregex)
    # train_it = dataset.train_batch(process_fn=precompute_wfn)
    train_it = dataset.train_batch()

    print p

    crashed = ['/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/piano/piano_0145_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/flower_pot/flower_pot_0056_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/chair/chair_0390_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/flower_pot/flower_pot_0109_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/stool/stool_0024_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/night_stand/night_stand_0025_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/tv_stand/tv_stand_0050_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/monitor/monitor_0389_visible_normals_bin.ply']
    crashed = ['/home/jbweibel/code/gronet/cmake_build/flower_pot_0109_visible_normals_bin.ply', '/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/flower_pot/flower_pot_0109_visible_normals_bin.ply']
    # crashed = ['/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/chair/chair_0320_visible_normals_bin.ply']
    crashed = ['/home/jbweibel/code/gronet/cmake_build/car_0047_visible_normals_bin_visible_normals_bin.ply']
    # crashed = ['/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0000_00/scene0000_00_vh_clean.ply']
    crashed = ['/home/jbweibel/dataset/ModelNet/ModelNet10_TrainPly/chair/chair_0001_visible_normals_bin.ply']
    crashed = ['/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0054_00/objects/table_14_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0056_00/objects/chair_1_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0056_00/objects/chair_3_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0057_00/objects/cabinet_12_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0057_00/objects/shelf_0_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0057_01/objects/shelf_3_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0059_00/objects/table_8_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0059_02/objects/chair_11_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0059_02/objects/chair_18_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0059_02/objects/chair_20_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0059_02/objects/table_26_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0068_01/objects/cabinet_4_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0073_01/objects/chair_16_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0102_00/objects/chair_2_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0102_01/objects/table_15_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0112_01/objects/cabinet_9_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0133_00/objects/chair_4_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0133_00/objects/chair_5_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0160_00/objects/lamp_14_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0160_04/objects/table lamp_4_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0160_04/objects/table_8_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0170_02/objects/bathtub_1_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0180_00/objects/chair_10_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0180_00/objects/chair_8_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0180_00/objects/table_0_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0198_00/objects/printer_1_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0237_00/objects/bathtub_0_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0240_00/objects/printer_0_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0248_01/objects/chair_6_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0264_01/objects/chair_6_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0331_00/objects/pillow_12_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0340_00/objects/table_15_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0409_00/objects/chair_10_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0416_00/objects/bed_2_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0416_00/objects/tv_16_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0416_01/objects/bed_26_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0416_01/objects/bedframe_4_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0416_02/objects/pillow_17_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0416_04/objects/pillow_32_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0483_00/objects/chair_18_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0495_00/objects/chair_19_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0495_00/objects/chair_20_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0505_00/objects/chair_16_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0505_00/objects/chair_19_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0505_02/objects/shelf_20_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0505_04/objects/table_29_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0506_00/objects/bedframe_1_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0529_01/objects/cabinet_9_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0539_00/objects/shelf_1_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0556_01/objects/bed_9_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0617_00/objects/table_5_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0617_00/objects/tv_7_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0630_03/objects/chair_9_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0630_05/objects/chair_21_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0630_05/objects/keyboard_8_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0630_06/objects/chair_10_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0642_00/objects/chair_9_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0642_02/objects/bedding set_5_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0642_02/objects/cabinet_4_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0706_00/objects/bed_3_wnormals.ply','/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/scans/scene0706_00/objects/bed_4_wnormals.ply']
    crashed = ["/home/jbweibel/dataset/ScanNet/scans/scene0152_02/objectsv2/chair_25_wnormals.ply",
               "/home/jbweibel/dataset/ScanNet/scans/scene0592_00/objectsv2/chair_19_wnormals.ply",
               "/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPly/chair/chair_0003_visible_normals_bin.ply"]
    crashed = ['scene0068_01/objectsv2/cup_33_wnormals.ply', 'scene0282_01/objectsv2/cup_15_wnormals.ply', 'scene0141_00/objectsv2/cup_60_wnormals.ply', 'scene0141_00/objectsv2/cup_61_wnormals.ply', 'scene0630_03/objectsv2/cup_52_wnormals.ply', 'scene0131_01/objectsv2/cup_30_wnormals.ply', 'scene0590_01/objectsv2/cup_28_wnormals.ply', 'scene0195_01/objectsv2/cup_29_wnormals.ply', 'scene0164_01/objectsv2/cup_33_wnormals.ply', 'scene0607_00/objectsv2/cup_13_wnormals.ply', 'scene0607_00/objectsv2/cup_14_wnormals.ply', 'scene0607_00/objectsv2/cup_15_wnormals.ply', 'scene0395_01/objectsv2/cup_2_wnormals.ply', 'scene0448_00/objectsv2/cup_15_wnormals.ply', 'scene0448_00/objectsv2/cup_17_wnormals.ply', 'scene0092_02/objectsv2/cup_38_wnormals.ply', 'scene0614_01/objectsv2/cup_17_wnormals.ply', 'scene0092_04/objectsv2/cup_14_wnormals.ply', 'scene0447_01/objectsv2/cup_14_wnormals.ply', 'scene0447_01/objectsv2/cup_12_wnormals.ply', 'scene0474_03/objectsv2/cup_20_wnormals.ply', 'scene0486_00/objectsv2/cup_36_wnormals.ply', 'scene0100_01/objectsv2/cup_16_wnormals.ply', 'scene0100_01/objectsv2/cup_18_wnormals.ply', 'scene0100_01/objectsv2/cup_15_wnormals.ply', 'scene0347_01/objectsv2/cup_32_wnormals.ply', 'scene0347_01/objectsv2/cup_29_wnormals.ply', 'scene0347_02/objectsv2/cup_30_wnormals.ply', 'scene0347_02/objectsv2/cup_28_wnormals.ply', 'scene0347_02/objectsv2/cup_29_wnormals.ply', 'scene0101_00/objectsv2/cup_61_wnormals.ply', 'scene0238_01/objectsv2/cup_17_wnormals.ply', 'scene0220_01/objectsv2/cup_19_wnormals.ply', 'scene0653_00/objectsv2/cup_49_wnormals.ply', 'scene0230_00/objectsv2/cup_29_wnormals.ply', 'scene0447_02/objectsv2/cup_14_wnormals.ply', 'scene0447_02/objectsv2/cup_11_wnormals.ply', 'scene0447_02/objectsv2/cup_15_wnormals.ply', 'scene0614_00/objectsv2/cup_35_wnormals.ply', 'scene0378_00/objectsv2/cup_31_wnormals.ply', 'scene0378_00/objectsv2/cup_32_wnormals.ply', 'scene0479_00/objectsv2/cup_9_wnormals.ply', 'scene0138_00/objectsv2/cup_0_wnormals.ply', 'scene0331_00/objectsv2/cup_14_wnormals.ply', 'scene0331_00/objectsv2/cup_16_wnormals.ply', 'scene0331_00/objectsv2/cup_15_wnormals.ply', 'scene0674_01/objectsv2/cup_12_wnormals.ply', 'scene0183_00/objectsv2/cup_6_wnormals.ply', 'scene0183_00/objectsv2/cup_5_wnormals.ply', 'scene0034_02/objectsv2/cup_21_wnormals.ply', 'scene0114_02/objectsv2/cup_17_wnormals.ply', 'scene0310_02/objectsv2/cup_32_wnormals.ply', 'scene0114_00/objectsv2/cup_26_wnormals.ply', 'scene0100_00/objectsv2/cup_18_wnormals.ply', 'scene0100_00/objectsv2/cup_19_wnormals.ply', 'scene0100_00/objectsv2/cup_3_wnormals.ply', 'scene0131_00/objectsv2/cup_23_wnormals.ply', 'scene0341_01/objectsv2/cup_20_wnormals.ply', 'scene0341_01/objectsv2/cup_37_wnormals.ply', 'scene0341_01/objectsv2/cup_16_wnormals.ply', 'scene0341_01/objectsv2/cup_15_wnormals.ply', 'scene0581_01/objectsv2/cup_4_wnormals.ply', 'scene0425_01/objectsv2/cup_26_wnormals.ply', 'scene0476_02/objectsv2/cup_21_wnormals.ply', 'scene0477_01/objectsv2/cup_23_wnormals.ply', 'scene0473_00/objectsv2/stack of cups_10_wnormals.ply', 'scene0473_00/objectsv2/stack of cups_9_wnormals.ply', 'scene0473_00/objectsv2/stack of cups_11_wnormals.ply', 'scene0117_00/objectsv2/cup_34_wnormals.ply', 'scene0117_00/objectsv2/cup_35_wnormals.ply', 'scene0117_00/objectsv2/cups_33_wnormals.ply', 'scene0630_04/objectsv2/cup_49_wnormals.ply', 'scene0126_00/objectsv2/cup_26_wnormals.ply', 'scene0268_00/objectsv2/cup_10_wnormals.ply', 'scene0268_00/objectsv2/cup_9_wnormals.ply', 'scene0498_00/objectsv2/cup_30_wnormals.ply', 'scene0498_00/objectsv2/cup_28_wnormals.ply', 'scene0322_00/objectsv2/cup_9_wnormals.ply', 'scene0151_01/objectsv2/cup_23_wnormals.ply', 'scene0034_01/objectsv2/cup_18_wnormals.ply', 'scene0474_01/objectsv2/cup_17_wnormals.ply', 'scene0309_01/objectsv2/cups_38_wnormals.ply', 'scene0653_01/objectsv2/cup_4_wnormals.ply', 'scene0464_00/objectsv2/cup_19_wnormals.ply', 'scene0464_00/objectsv2/cup_21_wnormals.ply', 'scene0282_00/objectsv2/cup_12_wnormals.ply', 'scene0198_00/objectsv2/cup_18_wnormals.ply', 'scene0058_01/objectsv2/cup_25_wnormals.ply', 'scene0498_02/objectsv2/cup_16_wnormals.ply', 'scene0498_02/objectsv2/cup_17_wnormals.ply', 'scene0225_00/objectsv2/cup_29_wnormals.ply', 'scene0341_00/objectsv2/cup_35_wnormals.ply', 'scene0341_00/objectsv2/cup_33_wnormals.ply', 'scene0341_00/objectsv2/cup_32_wnormals.ply', 'scene0282_02/objectsv2/cup_14_wnormals.ply', 'scene0614_02/objectsv2/cup_27_wnormals.ply', 'scene0040_00/objectsv2/cup_14_wnormals.ply', 'scene0040_00/objectsv2/cup_61_wnormals.ply', 'scene0040_00/objectsv2/cup_33_wnormals.ply', 'scene0370_00/objectsv2/cup_21_wnormals.ply']
    crashed = ["/home/jbweibel/dataset/ScanNet/scans/" + fn for fn in crashed]
    crashed = [[crashed_fn] for crashed_fn in crashed]

    # === PARAMETERS ==========================================================
    p.nodes_nb = 1
    p.neigh_size = 210.
    p.debug = True

    max_len = len(crashed)
    # max_len = dataset.train_batch_no
    if p.dataset == DATASETS.ScanNetToModelNet40:
        fn_repr = lambda fn: fn.split("/")[-3] + "/" + fn.split("/")[-2] + "/" + fn.split("/")[-1]
    else:
        fn_repr = lambda fn: fn.split("/")[-1]

    # for idx, (fns, yns) in enumerate(train_it):
    for idx, fns in enumerate(crashed[:5]):
        # log("{:.1f} % \n", (100. * idx / max_len))
        for fn in fns:
            fn_disp = fn_repr(fn)
            # fn_disp = fn.split("/")[-3] + "/" + fn.split("/")[-2] + "/" + fn.split("/")[-1]
            # fn_disp = fn.split("/")[-1]
            print "=", fn_disp, "="*(77 - len(fn_disp))
            graph = PyGraph(fn, nodes_nb=p.nodes_nb,
                            debug=p.debug,
                            gridsize=p.gridsize)

            # Initialization
            try:
                adj_mat = graph.initialize_mesh(p.angle_sampling, p.neigh_size)
                # node_feats = graph.node_features_coords_set(p.feat_nb[0], {"use_zlrf": True}, p.feat_nb[1])
                # tconv_idx, tconv_angle = graph.edge_features_tconv()
                # print tconv_idx[0].reshape((3, 3))
                # print tconv_angle[0].reshape((3, 3))
                # print adj_mat[0].nonzero()
                # graph.viz_graph(adj_mat, {"curvature": False,
                #                           "graph_skeleton": False,
                #                           "lrf": False,
                #                           "mesh": True,
                #                           "nodes": False,
                #                           "normals": False})
                # sampled_nodes = np.sum(graph.get_valid_indices())
                # viz_tconv_indices(tconv_idx, sampled_nodes)
                # viz_pcs(node_feats)
                # raw_input("Done with {}.\nPress ENTER to continue".format(fn_disp))
            except NanNormals as e:
                logw("NanNormals on {}", fn_disp)

            # sampled_nodes = np.sum(graph.get_valid_indices())
            # raw_input("Done with {}.\nPress ENTER to continue".format(
            #     fn.split("/")[-3] + "/" + fn.split("/")[-2] + "/" + fn.split("/")[-1]))
            # for i in range(sampled_nodes):
            #     viz_pc(node_feats[i])
