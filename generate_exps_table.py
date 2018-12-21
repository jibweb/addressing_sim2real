import argparse
import os
import pandas as pd
import yaml

OUTPUT_SAVE = "output_save"

KEYS_TO_DROP = [
    "debug", "gridsize",
    "noise_std", "occl_pct",  "rotation_deg"                     # AUGMENTATION
    "to_keep", "to_remove",                                      # AUGMENTATION
    "batch_size", "decay_rate", "decay_steps", "learning_rate",  # TRAINING
    "max_epochs", "reg_constant",                                # TRAINING
    "num_classes", "val_set_pct",                                # DATASET
    "attn_drop_prob", "feat_drop_prob", "pool_drop_prob",        # MODEL
    "residual", "transform",                                     # MODEL
    "viz", "viz_small_spheres",                                  # VIZ
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
            df = df.append(param_dict, ignore_index=True)

    if not args.no_drop:
        df = df.drop(columns=KEYS_TO_DROP)

    with open(OUTPUT_SAVE + "/exps_table.html", "w") as fp:
        with open(OUTPUT_SAVE + "/exps_header.html") as fhead:
            fp.write(fhead.read())
        fp.write(df.to_html(classes="display"))
        with open(OUTPUT_SAVE + "/exps_footer.html") as ffoot:
            fp.write(ffoot.read())
