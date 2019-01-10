import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

from dataset import get_dataset
from models import get_model
from preprocessing import get_graph_preprocessing_fn
from utils.logger import log, set_log_level, TimeScope
from utils.params import params as p
from utils.viz import plot_confusion_matrix

# ---- Parameters ----
# Generic
set_log_level("INFO")
TEST_REPEAT = 3


if __name__ == "__main__":
    # === SETUP ===============================================================
    # --- Parse arguments for specific params file ----------------------------
    parser = argparse.ArgumentParser("Test a given model on a given dataset")
    parser.add_argument("-f", "--exp_folder", help="Choose experiment folder")
    parser.add_argument("-c", "--model_ckpt", help="Choose model checkpoint")
    parser.add_argument("-v", "--viz", action="store_true",
                        help="Show Confusion Matrix")
    args = parser.parse_args()

    if args.exp_folder:
        SAVE_DIR = "output_save/" + args.exp_folder + "/"
    else:
        with open(".experiment_history") as fp:
            exp_folders = fp.readlines()
            # exp_folders = [exp.strip() for exp in exp_folders]
        SAVE_DIR = "output_save/" + exp_folders[-1].strip() + "/"

    os.system("rm -rf {}test_tb/*".format(SAVE_DIR))
    p.define_from_file("{}/params.yaml".format(SAVE_DIR))
    # p.rotation_deg = 180

    if args.model_ckpt:
        MODEL_CKPT = "model_{}/model.ckpt".format(args.model_ckpt)
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
    feat_compute = get_graph_preprocessing_fn(p)

    # --- Dataset setup -------------------------------------------------------
    Dataset, CLASS_DICT = get_dataset(p.dataset)
    regex = "/*_full_wnormals_wattention.ply" if p.mesh  \
            else "/*_full_wnormals_wattention.pcd"
    dataset = Dataset(batch_size=p.batch_size,
                      val_set_pct=p.val_set_pct,
                      regex=regex)

    # --- Model Setup ---------------------------------------------------------
    Model = get_model(p.model)
    model = Model()

    # --- Accuracy setup ------------------------------------------------------
    with tf.variable_scope('accuracy'):
        correct_prediction = tf.equal(
                tf.argmax(model.inference, 1),
                tf.argmax(model.y, 1))
        confusion = tf.confusion_matrix(
                labels=tf.argmax(model.y, 1),
                predictions=tf.argmax(model.inference, 1),
                num_classes=p.num_classes)

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
        total_cm = np.zeros((p.num_classes, p.num_classes), dtype=np.int32)

        for repeat in range(TEST_REPEAT):
            for xs, ys in dataset.test_batch(process_fn=feat_compute):
                with TimeScope("accuracy", debug_only=True):
                    summary, acc, loss, cm = sess.run(
                        [merged,
                         accuracy,
                         model.loss,
                         confusion],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=False))
                    test_writer.add_summary(summary, test_iter)

                total_acc = (test_iter*total_acc + acc) / (test_iter + 1)
                total_cm += cm

                log("Accurracy: {:.1f} / {:.1f} (loss: {:.3f})",
                    100.*total_acc,
                    100.*acc,
                    loss)

                test_iter += 1
        print ""
        confmat_filename = "{}conf_matrix_{}".format(SAVE_DIR,
                                                     MODEL_CKPT.split("/")[0])
        np.save(confmat_filename, total_cm)

        if args.viz:
            plot_confusion_matrix(total_cm, sorted(CLASS_DICT.keys()),
                                  normalize=True)
            plt.show()

        print total_cm
