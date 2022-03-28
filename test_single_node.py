#!/usr/bin/env python2.7

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

from dataset import get_dataset, DATASETS
from models import get_model
from preprocessing import get_graph_preprocessing_fn
from utils.logger import log, set_log_level, TimeScope
from utils.params import params as p
from utils.viz import plot_confusion_matrix

# ---- Parameters ----
# Generic
set_log_level("INFO")
TEST_REPEAT = 5


if __name__ == "__main__":

    p.debug = True

    # === SETUP ===============================================================
    # --- Parse arguments for specific params file ----------------------------
    parser = argparse.ArgumentParser("Test a given model on a given dataset")
    parser.add_argument("-f", "--exp_folder", help="Choose experiment folder")
    parser.add_argument("-c", "--model_ckpt", help="Choose model checkpoint")
    parser.add_argument("-d", "--dataset", help="Specify a different dataset")
    parser.add_argument("-v", "--viz", action="store_true",
                        help="Show Confusion Matrix")
    args = parser.parse_args()

    if args.exp_folder:
        # SAVE_DIR = "output_save/" + args.exp_folder + "/"
        SAVE_DIR = args.exp_folder + "/"
    else:
        with open(".experiment_history") as fp:
            exp_folders = fp.readlines()
            # exp_folders = [exp.strip() for exp in exp_folders]
        SAVE_DIR = "output_save/" + exp_folders[-1].strip() + "/"

    os.system("rm -rf {}test_tb/*".format(SAVE_DIR))
    p.define_from_file("{}/params.yaml".format(SAVE_DIR))
    # p.rotation_deg = 180

    if args.model_ckpt:
        MODEL_CKPT = "model_{0}/model.ckpt-{0}".format(args.model_ckpt)
    else:
        epochs_nb_found = sorted([int(dirname[6:])
                                  for dirname in os.listdir(SAVE_DIR)
                                  if dirname[:5] == "model"])
        if len(epochs_nb_found) != 0:
            MODEL_CKPT = "model_{0}/model.ckpt-{0}".format(epochs_nb_found[-1])
        else:
            log("Failed to find a trained model to restore\n")
            sys.exit()

    # --- Pre processing function setup ---------------------------------------
    p.neigh_size = 210
    feat_compute = get_graph_preprocessing_fn(p, with_fn=True)

    # --- Dataset setup -------------------------------------------------------
    different_test_dataset = False
    if args.dataset:
        different_test_dataset = True
        p.dataset = args.dataset

    Dataset, CLASS_DICT = get_dataset(p.dataset)
    pregex = "/*_visible_normals_bin.ply"
    if p.dataset == DATASETS.ModelNet10.name:
        pregex = "/*_full_wnormals_wattention.ply"
    elif p.dataset == DATASETS.ModelNet10PLY.name:
        pregex = "/*[0-9]_visible_normals_bin.ply"  # Manifold models
        # pregex = "/*_bin_visible_normals_bin.ply"  # TSDF models
    elif p.dataset == DATASETS.ScanNet.name:
        pregex = "*_wnormals.ply"
    dataset = Dataset(batch_size=p.batch_size,
                      val_set_pct=p.val_set_pct,
                      regex=pregex)

    # --- Model Setup ---------------------------------------------------------
    Model = get_model(p.model)
    model = Model()

    # --- Accuracy setup ------------------------------------------------------
    with tf.variable_scope('accuracy'):
        #--------- CHANGED -----------------------------------------------
        obj_inference = tf.reshape(model.inference, [-1, p.nodes_nb, p.num_classes])

        obj_mask = tf.reshape(model.mask, [-1, p.nodes_nb])
        obj_y = tf.reshape(model.y, [-1, p.nodes_nb, p.num_classes])

        inference = tf.boolean_mask(model.inference, model.mask)
        y = tf.boolean_mask(model.y, model.mask)

        # inference = tf.reduce_sum(inference, axis=1)
        # y = tf.reduce_max(y, axis=1)

        correct_prediction = tf.equal(
                tf.argmax(inference, 1),
                tf.argmax(y, 1))
        confusion = tf.confusion_matrix(
                labels=tf.argmax(y, 1),
                predictions=tf.argmax(inference, 1),
                num_classes=p.num_classes)
        #--------- /CHANGED -----------------------------------------------

        # correct_prediction = tf.equal(
        #         tf.argmax(model.inference, 1),
        #         tf.argmax(model.y, 1))
        # confusion = tf.confusion_matrix(
        #         labels=tf.argmax(model.y, 1),
        #         predictions=tf.argmax(model.inference, 1),
        #         num_classes=p.num_classes)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                          tf.float32))
        tf.summary.scalar('avg', accuracy)

    # --- Summaries and saver setup -------------------------------------------
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    # === GRAPH COMPUTATION ===================================================
    with tf.Session() as sess:
        saver.restore(sess, SAVE_DIR + MODEL_CKPT)

        # Summaries Writer
        test_writer = tf.summary.FileWriter(SAVE_DIR + 'test_tb')

        # Testing
        log("Setup finished, starting testing now ... \n\n")
        print "Parameters:"
        print p, "\n"
        test_iter = 0
        total_acc = 0.
        total_obj_acc = 0.
        total_cm = np.zeros((p.num_classes, p.num_classes), dtype=np.int32)

        obj_cls_preds = np.zeros((p.num_classes), dtype=float)
        obj_cls_count = np.zeros((p.num_classes), dtype=float)

        for repeat in range(TEST_REPEAT):
            for xs, ys in dataset.test_batch(process_fn=feat_compute):
                with TimeScope("accuracy", debug_only=True):
                    # summary, preds, acc, loss, cm = sess.run(
                    summary, preds, ys, masks, acc, cm = sess.run(
                        [merged,
                         obj_inference,
                         obj_y,
                         obj_mask,
                         accuracy,
                         confusion],
                        # correct_prediction,
                        # accuracy,
                        # model.loss,
                        # confusion],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=False))
                    test_writer.add_summary(summary, test_iter)

                x_fns = np.array([x_i[-1] for x_i in xs])

                masks = masks.astype(bool)
                batch_size = preds.shape[0]

                object_preds = []

                for b in range(batch_size):
                    # print preds[b][masks[b]].shape, np.sum(preds[b][masks[b]], axis=0).shape
                    is_correct = np.argmax(ys[b][0]) == np.argmax(np.sum(preds[b][masks[b]], axis=0))
                    object_preds.append(is_correct)
                    cls_idx = int(np.argmax(ys[b][0]))
                    obj_cls_preds[cls_idx] += float(is_correct)
                    obj_cls_count[cls_idx] += 1.
                    # print ys[b][masks[b]].shape, np.argmax(ys[b][masks[b]], axis=1).shape, np.unique(np.argmax(ys[b][masks[b]], axis=1))
                obj_acc = np.mean(object_preds)
                # acc = np.mean((np.argmax(ys, axis=1) == np.argmax(preds, axis=1)).astype(float))
                # print acc

                # misclassified_fns = x_fns[np.logical_not(preds)]

                total_acc = (test_iter*total_acc + acc) / (test_iter + 1)
                total_obj_acc = (test_iter*total_obj_acc + obj_acc) / (test_iter + 1)
                total_cm += cm

                log("Accurracy: {:.1f} / {:.1f} // Obj Acc. {:.1f} / {:.1f}",
                    100.*total_acc,
                    100.*acc,
                    100.*total_obj_acc,
                    100.*obj_acc)

                test_iter += 1

        print ""
        indices = np.logical_not(np.isnan(100.*total_cm.diagonal() / np.sum(total_cm, axis=1)))
        total_obj_indices = obj_cls_count != 0.
        total_obj_cls_acc = 100. * np.mean(obj_cls_preds[total_obj_indices] / obj_cls_count[total_obj_indices])
        print "Class Accuracy: {:.1f} // Obj class Accuracy: {:.1f}".format(
            np.mean((100.*total_cm.diagonal() / np.sum(total_cm, axis=1))[indices]),
            total_obj_cls_acc)

        CLASSES = np.array(sorted(CLASS_DICT.keys()))
        for cls_acc, cls_name in zip(obj_cls_preds[total_obj_indices] / obj_cls_count[total_obj_indices], CLASSES[total_obj_indices]):
            print "{}: {:.2f}".format(cls_name, 100. * cls_acc)

        if different_test_dataset:
            confmat_filename = "{}conf_matrix_{}_tested_on_{}".format(
                SAVE_DIR,
                MODEL_CKPT.split("/")[0],
                p.dataset)
        else:
            confmat_filename = "{}conf_matrix_{}".format(
                SAVE_DIR,
                MODEL_CKPT.split("/")[0])
        np.save(confmat_filename, total_cm)

        if args.viz:
            plot_confusion_matrix(total_cm, sorted(CLASS_DICT.keys()),
                                  normalize=True)
            plt.show()

        print total_cm
