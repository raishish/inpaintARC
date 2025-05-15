#!/usr/bin/env python
"""
Generate inpainting grid data from re-arc generated input-output pairs.

Usage: python3 gen_inpaint_data.py --combined_dir all_pairs --train_dir train --val_dir val --test_dir test \
       --train_ratio 0.8 --val_ratio 0.04 --test_ratio 0.16

We generate 10,000 input-output pairs from re-arc per task.
We fix the number of training examples per task to 3 (average example tasks in the ARC dataset).
"""

import glob
import os
import json
from argparse import ArgumentParser

NUM_TRAIN_EXAMPLES = 3
NUM_TEST_EXAMPLES = 1


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Split all re-arc generated input-output pairs into train/validation/test sets."
    )
    parser.add_argument(
        "--combined_dir",
        type=str,
        default="data/rearc/all_pairs",
        help="Directory containing all pairs."
    )
    parser.add_argument("--train_dir", type=str, default="data/rearc/train", help="Directory to save training pairs.")
    parser.add_argument("--val_dir", type=str, default="data/rearc/val", help="Directory to save validation pairs.")
    parser.add_argument("--test_dir", type=str, default="data/rearc/test", help="Directory to save test pairs.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data.")
    parser.add_argument("--val_ratio", type=float, default=0.04, help="Ratio of validation data.")
    parser.add_argument("--test_ratio", type=float, default=0.16, help="Ratio of test data.")
    args = parser.parse_args()

    # glob for all jsonl files in the combined directory
    files = glob.glob(os.path.join(args.combined_dir, "*.jsonl"))
    print(f"Found {len(files)} files in {args.combined_dir}.")

    for file in files:
        # Load the jsonl file
        all_pairs = []
        with open(file, "r") as f:
            for line in f:
                all_pairs.append(json.loads(line))

        train_pairs = all_pairs[:int(len(all_pairs) * args.train_ratio)]
        val_pairs = all_pairs[
            int(len(all_pairs) * args.train_ratio):int(len(all_pairs) * (args.train_ratio + args.val_ratio))
        ]
        test_pairs = all_pairs[int(len(all_pairs) * (args.train_ratio + args.val_ratio)):]

        # create train dataset
        train_tasks = []
        for i in range(0, len(train_pairs), NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES):
            task = {
                "train": [{"input": pair["input"], "output": pair["output"]}
                          for pair in train_pairs[i:i + NUM_TRAIN_EXAMPLES]],
                "test": [{"input": pair["input"], "output": pair["output"]}
                         for pair in train_pairs[i + NUM_TRAIN_EXAMPLES:i + NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES]]
            }
            train_tasks.append(task)

        # create val dataset
        val_tasks = []
        for i in range(0, len(val_pairs), NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES):
            task = {
                "train": [{"input": pair["input"], "output": pair["output"]}
                          for pair in val_pairs[i:i + NUM_TRAIN_EXAMPLES]],
                "test": [{"input": pair["input"], "output": pair["output"]}
                         for pair in val_pairs[i + NUM_TRAIN_EXAMPLES:i + NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES]]
            }
            val_tasks.append(task)

        # create test dataset
        test_tasks = []
        for i in range(0, len(test_pairs), NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES):
            task = {
                "train": [{"input": pair["input"], "output": pair["output"]}
                          for pair in test_pairs[i:i + NUM_TRAIN_EXAMPLES]],
                "test": [{"input": pair["input"], "output": pair["output"]}
                         for pair in test_pairs[i + NUM_TRAIN_EXAMPLES:i + NUM_TRAIN_EXAMPLES + NUM_TEST_EXAMPLES]]
            }
            test_tasks.append(task)

        print(f"Saving {len(train_tasks)} training tasks, " +
              f"{len(val_tasks)} validation tasks, and {len(test_tasks)} test tasks.")

        # save train dataset
        with open(os.path.join(args.train_dir, file.split("/")[-1]), "w") as f:
            for task in train_tasks:
                f.write(json.dumps(task) + "\n")

        # save val dataset
        with open(os.path.join(args.val_dir, file.split("/")[-1]), "w") as f:
            for task in val_tasks:
                f.write(json.dumps(task) + "\n")

        # save test dataset
        with open(os.path.join(args.test_dir, file.split("/")[-1]), "w") as f:
            for task in test_tasks:
                f.write(json.dumps(task) + "\n")