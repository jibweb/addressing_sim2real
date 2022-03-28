from functools import partial
from dataset import get_dataset, DATASETS
import os
from utils.logger import log

EXTRACTOR_BIN_DIR = "/home/jbweibel/code/gronet/cmake_build/"
# MESHING_EXE = EXTRACTOR_BIN_DIR + "meshing"
# VIEW_EXTRACTOR_EXE = EXTRACTOR_BIN_DIR + "view_extractor"
# NORM_PRECOMP_EXE = EXTRACTOR_BIN_DIR + "normal_precomputation"
CLEANUP_OFF_EXE = EXTRACTOR_BIN_DIR + "cleanup_off"
BINARIZE_PLY_EXE = EXTRACTOR_BIN_DIR + "binarize_ply"
BIN_VISIBLE_NORMALS_EXE = EXTRACTOR_BIN_DIR + "remove_insides_norm_bin"

# --- ModelNet 10 -------------------------------------------------------------
MN_DIR = "/home/jbweibel/dataset/ModelNet/"
MN10_SAVE_DIR_TRAIN = MN_DIR + "ModelNet10_TrainPc/"
MN10_SAVE_DIR_TEST = MN_DIR + "ModelNet10_TestPc/"

MN10OFF_SAVE_DIR_TRAIN = MN_DIR + "ModelNet10_TrainPly/"
MN10OFF_SAVE_DIR_TEST = MN_DIR + "ModelNet10_TestPly/"

# --- ModelNet 40 -------------------------------------------------------------
MN40_SAVE_DIR_TRAIN = MN_DIR + "modelnet40_manually_aligned_TrainPly/"
MN40_SAVE_DIR_TEST = MN_DIR + "modelnet40_manually_aligned_TestPly/"


def precompute(in_fn, out_dir, exe):
    # print in_fn
    # For ModelNet10
    # cls_name = in_fn.split("/")[-2] + "/"

    # For ModelNet10OFF
    cls_name = in_fn.split("/")[-3] + "/"
    obj_name = in_fn.split("/")[-1].split(".")[0]

    # OFF to OBJ
    os.system("off2obj {} > {}.obj".format(in_fn,
                                           out_dir+cls_name+obj_name))

    # Manifold
    MANIFOLD_EXE = "/home/jbweibel/code/Manifold/build/manifold"
    # Lower res for MN10
    os.system("{} {}.obj {}.obj 8000".format(MANIFOLD_EXE,
                                             out_dir+cls_name+obj_name,
                                             out_dir+cls_name+obj_name))
    # Higher res for MN40
    # os.system("{} {}.obj {}.obj 15000".format(MANIFOLD_EXE,
    #                                           out_dir+cls_name+obj_name,
    #                                           out_dir+cls_name+obj_name))

    # Take the subset, compute the normals and make a binary
    os.system("{} -i {}.obj -o {}".format(exe, out_dir+cls_name+obj_name,
                                          out_dir+cls_name))
    print out_dir+cls_name+obj_name

    # For ScanNet
    # in_fn = '\\ '.join(in_fn.split(" "))
    # out_dir = "/".join(in_fn.split("/")[:-1]) + "/"

    # For S3DIS
    # in_fn = in_fn[:-4] + ".pcd"
    # out_dir = "/".join(in_fn.split("/")[:-1]) + "/"
    # os.system("{} -i {} -o {}".format(exe, in_fn, out_dir))

    return in_fn


if __name__ == "__main__":
    # DATASET = DATASETS.ModelNet10PLY
    DATASET = DATASETS.ModelNet10OFF
    Dataset, CLASS_DICT = get_dataset(DATASET)
    dataset = Dataset(# regex="/*_bin.ply",
                      balance_train_set=False,
                      balance_test_set=False,
                      batch_size=6,
                      val_set_pct=0.)
    executable = BIN_VISIBLE_NORMALS_EXE

    mn10ply_train = partial(precompute, exe=executable,
                            out_dir=MN10OFF_SAVE_DIR_TRAIN)
    mn10ply_test = partial(precompute, exe=executable,
                           out_dir=MN10OFF_SAVE_DIR_TEST)

    # mn40ply_train = partial(precompute, exe=executable,
    #                         out_dir=MN40_SAVE_DIR_TRAIN)
    # mn40ply_test = partial(precompute, exe=executable,
    #                        out_dir=MN40_SAVE_DIR_TEST)
    # scannet_all = partial(precompute, exe=executable,
    #                       out_dir="")

    train_it = dataset.train_batch(process_fn=mn10ply_train, timeout=120)
    test_it = dataset.test_batch(process_fn=mn10ply_test, timeout=120)

    for idx, (fns, yns) in enumerate(train_it):
        log("{:.1f} % ", 100.*idx/dataset.train_batch_no)

    print "\n\nFINISHED TRAIN SET\n\n"

    for idx, (fns, yns) in enumerate(test_it):
        log("{:.1f} % ", 100.*idx/dataset.test_batch_no)

    print "\n\nFINISHED TEST SET\n\n"

    # if DATASET == DATASETS.ScanNet:
    #     val_it = dataset.val_batch(process_fn=scannet_all)
    #     for idx, (fns, yns) in enumerate(val_it):
    #         log("{:.1f} % ", 100.*idx/dataset.val_batch_no)
