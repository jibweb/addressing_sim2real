#!/usr/bin/env python2.7

import argparse
import dfgui
from glob import glob
import numpy as np
import os
import pandas as pd
import yaml

OUTPUT_SAVE = "output_save"

KEYS_TO_DROP = [
    "debug", "gridsize",
    "noise_std", "occl_pct",  "rotation_deg",                    # AUGMENTATION
    "to_keep", "to_remove",                                      # AUGMENTATION
    "batch_size", "decay_rate", "decay_steps", "learning_rate",  # TRAINING
    "max_epochs", "reg_constant",                                # TRAINING
    "num_classes", "val_set_pct",                                # DATASET
    "attn_drop_prob", "feat_drop_prob", "pool_drop_prob",        # MODEL
    "residual",                                                  # MODEL
    "viz", "viz_small_spheres",                                  # VIZ
    "should_hash"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test a given model on a given dataset")
    parser.add_argument('--no_drop', action='store_true', default=False)
    args = parser.parse_args()

    df = pd.DataFrame()
    for directory in os.listdir(OUTPUT_SAVE):
        path = "{}/{}".format(OUTPUT_SAVE, directory)
        if os.path.isdir(path):
            with open(path + "/params.yaml") as fp:
                param_dict = yaml.load(fp)

            acc_results = []
            class_acc_results = []
            for result in sorted(glob(path + "/conf_matrix_model_*")):
                print result
                conf_mat = np.load(result)
                epoch_nb = result[:-4].split("/")[-1].split("_")[-1]
                acc = 100. * np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
                acc_results.append("{:.2f}@{}".format(acc, epoch_nb))

                class_acc = np.mean(100.*conf_mat.diagonal() / np.sum(conf_mat, axis=1))
                class_acc_results.append("{:.2f}@{}".format(class_acc, epoch_nb))

            models_found = sorted([int(dirname[6:])
                                  for dirname in os.listdir(path)
                                  if dirname[:5] == "model"])
            param_dict["trained_until"] = models_found[-1] \
                if len(models_found) != 0 else 0

            param_dict["accuracy"] = ", ".join(acc_results)
            param_dict["class_accuracy"] = ", ".join(class_acc_results)
            df = df.append(param_dict, ignore_index=True)

    if not args.no_drop:
        df = df.drop(columns=KEYS_TO_DROP)

    dfgui.show(df)
