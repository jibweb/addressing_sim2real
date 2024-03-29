#!/usr/bin/env python2.7

import argparse
from dataset import get_dataset, DATASETS
import os
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_mean

from preprocessing import get_graph_preprocessing_fn
from models import get_model, MODELS
from progress.bar import Bar
from utils.logger import log, logd, set_log_level, TimeScope
from utils.params import params as p


# === PARAMETERS ==============================================================
p.define("dataset", DATASETS.ModelNet40.name)
p.define("num_classes", 40)
p.define("model", MODELS.EFA_CoolPool.name)
# Training parameters
p.define("max_epochs", 500, should_hash=False)
p.define("batch_size", 32)
p.define("learning_rate", 0.001)
p.define("reg_constant", 0.01)
p.define("decay_steps", 10000)
p.define("decay_rate", 0.96)
p.define("val_set_pct", 0.1)

p.define("comment", "")

# Generic
set_log_level("INFO")

DEFAULT_PARAMS_FILE = "params/{}_{}.yaml".format(p.dataset,
                                                 p.model)

if __name__ == "__main__":
    # === SETUP ===============================================================
    # --- Parse arguments for specific params file ----------------------------
    parser = argparse.ArgumentParser("Test a given model on a given dataset")
    parser.add_argument("--params", type=str, help="params file to load",
                        default=DEFAULT_PARAMS_FILE)
    args = parser.parse_args()
    p.load(args.params)

    Dataset, CLASS_DICT = get_dataset(p.dataset)
    Model = get_model(p.model)

    p.num_classes = len(CLASS_DICT)

    EXPERIMENT_VERSION = p.get_hash()
    EXP_FOLDER = "{}_{}_{}".format(p.dataset,
                                   p.model,
                                   EXPERIMENT_VERSION)
    SAVE_DIR = "output_save/{}/".format(EXP_FOLDER)
    os.system("mkdir -p " + SAVE_DIR)

    # --- Clean/Restore previous experiments logs -----------------------------
    restore = False
    start_epoch = 1
    if len(os.listdir(SAVE_DIR)) != 0:
        log("save dir: {}\n", SAVE_DIR)
        if raw_input("A similar experiment was already saved."
                     " Would you like to delete it ?") in ['y', 'yes', 'Y']:
            os.system("rm -rf {}/*".format(SAVE_DIR))
            log("Experiment deleted !\n")
        else:
            epochs_nb_found = sorted([int(dirname[6:])
                                      for dirname in os.listdir(SAVE_DIR)
                                      if dirname[:5] == "model"])
            if len(epochs_nb_found) != 0:
                log("Restoring the records, tensorboard might be screwed up\n")
                restore = True
                start_epoch = epochs_nb_found[-1] + 1
                last_ckpt = "model_{0}/model.ckpt-{0}".format(epochs_nb_found[-1])
            else:
                log("Failed to find a trained model to restore\n")

    # --- Keep track of the latest experiments and parameters -----------------
    p.define("exp_id", EXPERIMENT_VERSION)
    p.save(SAVE_DIR + "params.yaml")

    with open(".experiment_history") as fp:
        exp_folders = fp.readlines()
        exp_folders = [exp.strip() for exp in exp_folders]

    exp_folders.append(EXP_FOLDER)

    with open(".experiment_history", "w") as fp:
        for exp in exp_folders[-50:]:
            fp.write(exp + "\n")

    with TimeScope("setup", debug_only=True):
        # --- Pre processing function setup -----------------------------------
        feat_compute = get_graph_preprocessing_fn(p)

        # --- Dataset setup ---------------------------------------------------
        pregex = "/*_visible_normals_bin.ply"
        if p.dataset == DATASETS.ModelNet10.name:
            pregex = "/*_full_wnormals_wattention.ply"
        elif p.dataset == DATASETS.ModelNet10PLY.name:
            pregex = "/*[0-9]_visible_normals_bin.ply"  # Manifold models
            # pregex = "/*_bin_visible_normals_bin.ply"  # TSDF models
        elif p.dataset == DATASETS.ScanNet.name:
            pregex = "*_wnormals.ply"
        pbalance_train_set = True  # False if p.dataset == DATASETS.ScanNet.name \
        #    else True
        dataset = Dataset(batch_size=p.batch_size,
                          balance_train_set=pbalance_train_set,
                          val_set_pct=p.val_set_pct,
                          regex=pregex)

        # --- Model Setup -----------------------------------------------------
        model = Model()

        # --- Accuracy setup --------------------------------------------------
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(
                    tf.argmax(model.inference, 1),
                    tf.argmax(model.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32))
            tf.summary.scalar('avg', accuracy)

        # --- Optimisation Setup ----------------------------------------------
        batch = tf.Variable(0, trainable=False, name="step")
        learning_rate = tf.train.exponential_decay(p.learning_rate,
                                                   batch,
                                                   p.decay_steps,
                                                   p.decay_rate,
                                                   staircase=True)
        # Clip the learning rate !
        learning_rate = tf.maximum(learning_rate, 0.00001)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(model.loss,
                      global_step=batch,
                      name="optimizer")

        # --- Summaries and saver setup ---------------------------------------
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=20)

    # === GRAPH COMPUTATION ===================================================
    with tf.Session() as sess:
        # Summaries Writer
        train_writer = tf.summary.FileWriter(SAVE_DIR + 'train_tb',
                                             sess.graph)
        val_writer = tf.summary.FileWriter(SAVE_DIR + 'val_tb')

        # Streaming summaries for validation set
        saccuracy, saccuracy_update = streaming_mean(accuracy)
        sloss, sloss_update = streaming_mean(model.loss)
        with tf.variable_scope('accuracy/'):
            saccuracy_scalar = tf.summary.scalar('val_accuracy',
                                                 saccuracy)
        with tf.variable_scope('loss/'):
            sloss_scalar = tf.summary.scalar('val_loss',
                                             sloss)

        if restore:
            saver.restore(sess, SAVE_DIR + last_ckpt)
        else:
            # Init
            sess.run(tf.global_variables_initializer())

        # Training
        log("Setup finished, starting training now ... \n\n")
        print "Parameters:"
        print p, "\n"
        print "pregex", pregex
        print "pbalance_train_set", pbalance_train_set

        for epoch in range(start_epoch, p.max_epochs+1):
            # --- Training step -----------------------------------------------
            train_iterator = dataset.train_batch(process_fn=feat_compute)
            bar_name = "Epoch {}/{}".format(epoch, p.max_epochs)
            bar = Bar(bar_name, max=dataset.train_batch_no)
            for idx, (xs, ys) in enumerate(train_iterator):
                with TimeScope("optimize", debug_only=True):
                    summaries, loss, _ = sess.run(
                        [merged,
                         model.loss,
                         train_op],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=True))
                train_iter = idx + (epoch-1)*dataset.train_batch_no
                train_writer.add_summary(summaries, train_iter)
                bar.next()
            bar.finish()

            # --- Validation accuracy -----------------------------------------
            # Re initialize the streaming means
            sess.run(tf.local_variables_initializer())

            for xs, ys in dataset.val_batch(process_fn=feat_compute):
                with TimeScope("validation", debug_only=True):
                    sess.run(
                        [saccuracy_update, sloss_update],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=False))
            saccuracy_summ, cur_sacc, sloss_summ, cur_sloss = sess.run(
                [saccuracy_scalar, saccuracy, sloss_scalar, sloss])

            cur_iter = (epoch-1)*dataset.train_batch_no
            val_writer.add_summary(saccuracy_summ, cur_iter)
            val_writer.add_summary(sloss_summ, cur_iter)

            log("Epoch {}/{} | Accuracy: {:.1f} (loss: {:.3f})\n",
                epoch,
                p.max_epochs,
                100.*cur_sacc,
                cur_sloss)

            # --- Save the model ----------------------------------------------
            if epoch % 10 == 0:
                # Save the variables to disk.
                save_path = saver.save(
                    sess,
                    "{}model_{}/model.ckpt".format(SAVE_DIR, epoch),
                    global_step=epoch)
                logd("Model saved in file: {}", save_path)

    dataset.close()
