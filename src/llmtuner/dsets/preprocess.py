import os
import numpy as np
import re
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union
from datasets import load_from_disk
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)


def pad_sequence(lists, padding_value, cut_len):
    """Pad all input sequences to the same length."""
    new_lists = []
    max_len = max([len(each_list) for each_list in lists])
    # if max_len < cut_len:
    #     cut_len = max_len

    logger.info(f"max_len: {max_len}")
    logger.info(f"Padding all sequences to length {cut_len}.")
    for each_list in lists:
        if len(each_list) >= cut_len:
            new_lists.append(each_list[:cut_len])
        else:
            new_lists.append(each_list+[padding_value]*(cut_len-len(each_list)))
    return new_lists


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: str,
) -> Union["Dataset", "IterableDataset"]:

    """
    def preprocess_s2s_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "src_mask": []}

        for src, tgt in zip(examples['prompt'], examples['response']):
            src_ids = tokenizer.encode(src)
            tgt_ids = tokenizer.encode(tgt)
            if data_args.cutoff_len is not None:
                tgt_ids = tgt_ids[:(data_args.cutoff_len-2)]
                src_ids = src_ids[-(data_args.cutoff_len-2-len(tgt_ids)):]

            input_ids = src_ids + [tokenizer.sep_token_id] + tgt_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["src_mask"].append([1] * (len(src_ids) + 1))

        model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], padding_value=tokenizer.pad_token_id, cut_len=data_args.cutoff_len)
        model_inputs["attention_mask"] = pad_sequence(model_inputs["attention_mask"], padding_value=1, cut_len=data_args.cutoff_len)
        model_inputs["src_mask"] = pad_sequence(model_inputs["src_mask"], padding_value=0, cut_len=data_args.cutoff_len)
        # print(model_inputs)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def print_unsupervised_dataset_example(example):
        # split by <arc_sep_row>
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    """

    def pad_grid(grid: List[List[int]], max_x: int, max_y: int, test_output: str = False) -> List[int]:
        """Pad an input/output grid with maximum dimensions."""
        arc_grid_endx_token = "<arc_grid_endx>"
        arc_grid_endy_token = "<arc_grid_endy>"
        arc_grid_endxy_token = "<arc_grid_endxy>"
        arc_sep_row_token = "<arc_sep_row>"
        arc_pad_token = "<arc_pad>"
        mask_token = "<mask>"

        orig_x, orig_y = len(grid[0]), len(grid)
        required_padding_x = max_x - orig_x
        required_padding_y = max_y - orig_y

        padded_rows = []
        for row in grid:
            if test_output:
                padded_row = [mask_token] * orig_x + [arc_grid_endx_token] + \
                    [arc_pad_token] * required_padding_x + [arc_sep_row_token]
            else:
                padded_row = row + [arc_grid_endx_token] + [arc_pad_token] * required_padding_x + [arc_sep_row_token]
            padded_rows.append(padded_row)

        # y boundary
        padded_rows.append(
            [arc_grid_endy_token] * orig_x +
            [arc_grid_endxy_token] +
            [arc_grid_endy_token] * required_padding_x +
            [arc_sep_row_token]
        )

        for row in range(required_padding_y):
            padded_rows.append(
                [arc_pad_token] * orig_x +
                [arc_grid_endx_token] +
                [arc_pad_token] * required_padding_x +
                [arc_sep_row_token]
            )

        return padded_rows

    def preprocess_masked_diffusion_dataset(tasks: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Preprocess the ARC dataset.
        Args:
            tasks (Dict[str, List[Any]]): batched tasks with train and test examples.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the dataset.
        Returns:
            Dict[str, Any]: The preprocessed dataset.
        """
        model_inputs = {"input_ids": [], "attention_mask": [], "src_mask": []}

        arc_sep_example_token = "<arc_sep_example>"
        arc_sep_grid_token = "<arc_sep_grid>"

        logger.info(f"Loaded a batch of {len(tasks['train'])} tasks.")
        # logger.info(f"first task training examples: {tasks['train'][0]}")
        for train_examples, test_examples in zip(tasks["train"], tasks["test"]):
            # train_examples = task["train"]
            # test_examples = task["test"]

            max_input_x = max([len(ex["input"][0]) for ex in train_examples + test_examples])
            max_output_x = max([len(ex["output"][0]) for ex in train_examples + test_examples])

            inpaint_grid = []
            for idx, ex in enumerate(train_examples):
                # if idx != 0:
                #     # separate each example with <arc_sep_example>
                #     inpaint_grid.append(np.array([arc_sep_example_token] * inpaint_grid[0].shape[1]))

                input_grid = ex["input"]
                output_grid = ex["output"]

                max_y = max(len(input_grid), len(output_grid))

                padded_input = pad_grid(input_grid, max_input_x, max_y, test_output=False)
                padded_output = pad_grid(output_grid, max_output_x, max_y, test_output=False)

                assert len(padded_input) == len(padded_output), \
                    "Padded input and output grids must have the same number of rows."
                assert len(padded_input[0]) == len(padded_output[0]), \
                    "Padded input and output grids must have the same number of columns."

                padded_input_np = np.array(padded_input)
                padded_output_np = np.array(padded_output)

                boundary = np.array([arc_sep_grid_token] * padded_input_np.shape[0])
                padded_example = np.concatenate((padded_input_np, boundary[:, None], padded_output_np), axis=1)

                inpaint_grid.append(padded_example)

            # add test examples
            for idx, ex in enumerate(test_examples):
                # inpaint_grid.append([arc_sep_example_token] * inpaint_grid[0].shape[0])

                input_grid = ex["input"]
                output_grid = ex["output"]

                max_y = max(len(input_grid), len(output_grid))

                padded_input = pad_grid(input_grid, max_input_x, max_y, test_output=False)
                padded_output = pad_grid(output_grid, max_output_x, max_y, test_output=True)

                assert len(padded_input) == len(padded_output), \
                    "Padded input and output grids must have the same number of rows."
                assert len(padded_input[0]) == len(padded_output[0]), \
                    "Padded input and output grids must have the same number of columns."

                padded_input_np = np.array(padded_input)
                padded_output_np = np.array(padded_output)

                boundary = np.array([arc_sep_grid_token] * padded_input_np.shape[0])
                padded_example = np.concatenate((padded_input_np, boundary[:, None], padded_output_np), axis=1)

                inpaint_grid.append(padded_example)

            example_sep_boundary = arc_sep_example_token * inpaint_grid[0].shape[1]
            inpaint_grid = [padded_example.flatten().tolist() for padded_example in inpaint_grid]
            inpaint_grid = [''.join(padded_example_flattened) for padded_example_flattened in inpaint_grid]
            inpaint_grid_str = example_sep_boundary.join(inpaint_grid)

            # inpaint_grid_flattened = inpaint_grid.flatten().tolist()
            # inpaint_grid_str = "".join(inpaint_grid_flattened)
            inpaint_grid_tok = tokenizer.encode(inpaint_grid_str)
            model_inputs["input_ids"].append(inpaint_grid_tok)
            model_inputs["attention_mask"].append([1] * len(inpaint_grid_tok))
            model_inputs["src_mask"].append([1] * (len(inpaint_grid_tok) - len(inpaint_grid[-1])))

        model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], padding_value=tokenizer.pad_token_id, cut_len=data_args.cutoff_len)
        model_inputs["attention_mask"] = pad_sequence(model_inputs["attention_mask"], padding_value=1, cut_len=data_args.cutoff_len)
        model_inputs["src_mask"] = pad_sequence(model_inputs["src_mask"], padding_value=0, cut_len=data_args.cutoff_len)
        return model_inputs

    def print_masked_diffusion_example(example):
        """Print an example from the ARC dataset."""
        # Extract the input_ids, attention_mask, and src_mask from the example
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        src_mask = example["src_mask"]
        # Decode the input_ids to get the original string
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        # Print the decoded input
        # print("Decoded input:\n{}".format(decoded_input))
        # regex match to split the decoded input by a continuous string of repeated <arc_sep_example>
        # split_input = re.split(r"(<arc_sep_example>)+", decoded_input)
        split_input = re.split(r"(<arc_sep_row>)+", decoded_input)
        # Print the split input
        for i, example in enumerate(split_input):
            print("Example {}:\n{}".format(i, example))
        # Attention mask
        print("Attention mask:\n{}".format(attention_mask))
        # Source mask
        print("Source mask:\n{}".format(src_mask))

    def preprocess_ar_supervised_dataset(tasks: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Preprocess the ARC dataset.
        Args:
            tasks (Dict[str, List[Any]]): batched tasks with train and test examples.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the dataset.
        Returns:
            Dict[str, Any]: The preprocessed dataset.
        """
        model_inputs = {"input_ids": [], "attention_mask": [], "src_mask": []}

        arc_sep_example_token = "<arc_sep_example>"
        arc_sep_grid_token = "<arc_sep_grid>"

        for task in tasks:
            train_examples = task["train"]
            test_examples = task["test"]

            max_input_x = max([len(ex["input"][0]) for ex in train_examples + test_examples])
            max_output_x = max([len(ex["output"][0]) for ex in train_examples + test_examples])

            inpaint_grid = []
            for idx, ex in enumerate(train_examples):
                if idx != 0:
                    # separate each example with <arc_sep_example>
                    inpaint_grid.append([arc_sep_example_token] * len(inpaint_grid[0]))

                input_grid = ex["input"]
                output_grid = ex["output"]

                max_y = max(len(input_grid), len(output_grid))

                padded_input = pad_grid(input_grid, max_input_x, max_y, test_output=False)
                padded_output = pad_grid(output_grid, max_output_x, max_y, test_output=False)

                assert len(padded_input) == len(padded_output), \
                    "Padded input and output grids must have the same number of rows."
                assert len(padded_input[0]) == len(padded_output[0]), \
                    "Padded input and output grids must have the same number of columns."

                padded_input_np = np.array(padded_input)
                padded_output_np = np.array(padded_output)

                boundary = np.array([arc_sep_grid_token] * padded_input_np.shape[0])
                padded_example = np.concatenate((padded_input_np, boundary[:, None], padded_output_np), axis=1)

                inpaint_grid.append(padded_example)

            # add test examples
            for idx, ex in enumerate(test_examples):
                input_grid = ex["input"]
                output_grid = ex["output"]

                max_y = max(len(input_grid), len(output_grid))

                padded_input = pad_grid(input_grid, max_input_x, max_y, test_output=False)
                padded_output = pad_grid(output_grid, max_output_x, max_y, test_output=True)

                assert len(padded_input) == len(padded_output), \
                    "Padded input and output grids must have the same number of rows."
                assert len(padded_input[0]) == len(padded_output[0]), \
                    "Padded input and output grids must have the same number of columns."

                padded_input_np = np.array(padded_input)
                padded_output_np = np.array(padded_output)

                boundary = np.array([arc_sep_grid_token] * padded_input_np.shape[0])
                padded_example = np.concatenate((padded_input_np, boundary[:, None], padded_output_np), axis=1)

                inpaint_grid.append(padded_example)

            inpaint_grid_flattened = inpaint_grid.flatten().tolist()
            inpaint_grid_str = "".join(inpaint_grid_flattened)
            inpaint_grid_tok = tokenizer.encode(inpaint_grid_str)
            model_inputs["input_ids"].append(inpaint_grid_tok)
            model_inputs["attention_mask"].append([1] * len(inpaint_grid_tok))
            model_inputs["src_mask"].append([1] * (len(inpaint_grid_flattened) + 1))

        return model_inputs

    def print_ar_supervised_example(input_ids: List[int], attention_mask: List[int], src_mask: List[int]):
        """Print an example from the ARC dataset."""
        # Decode the input_ids to get the original string
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        # Print the decoded input
        print("Decoded input:\n{}".format(decoded_input))
        # regex match to split the decoded input by a continuous string of repeated <arc_sep_example>
        split_input = re.split(r"(<arc_sep_example>)+", decoded_input)
        # Print the split input
        for i, part in enumerate(split_input):
            print("Example {}:\n{}".format(i, part))

    def preprocess_ar_unsupervised_dataset(tasks: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Preprocess the ARC dataset.
        Args:
            tasks (Dict[str, List[Any]]): batched tasks with train and test examples.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the dataset.
        Returns:
            Dict[str, Any]: The preprocessed dataset.
        """
        model_inputs = {"input_ids": [], "attention_mask": [], "src_mask": []}

        arc_sep_example_token = "<arc_sep_example>"
        arc_sep_grid_token = "<arc_sep_grid>"

        for task in tasks:
            train_examples = task["train"]
            test_examples = task["test"]

            max_input_x = max([len(ex["input"][0]) for ex in train_examples + test_examples])
            max_output_x = max([len(ex["output"][0]) for ex in train_examples + test_examples])

            inpaint_grid = []
            for idx, ex in enumerate(train_examples):
                if idx != 0:
                    # separate each example with <arc_sep_example>
                    inpaint_grid.append([arc_sep_example_token] * len(inpaint_grid[0]))

                input_grid = ex["input"]
                output_grid = ex["output"]

                max_y = max(len(input_grid), len(output_grid))

                padded_input = pad_grid(input_grid, max_input_x, max_y, test_output=False)
                padded_output = pad_grid(output_grid, max_output_x, max_y, test_output=False)

                assert len(padded_input) == len(padded_output), \
                    "Padded input and output grids must have the same number of rows."
                assert len(padded_input[0]) == len(padded_output[0]), \
                    "Padded input and output grids must have the same number of columns."

                padded_input_np = np.array(padded_input)
                padded_output_np = np.array(padded_output)

                boundary = np.array([arc_sep_grid_token] * padded_input_np.shape[0])
                padded_example = np.concatenate((padded_input_np, boundary[:, None], padded_output_np), axis=1)

                inpaint_grid.append(padded_example)

            # add test examples
            for idx, ex in enumerate(test_examples):
                input_grid = ex["input"]
                output_grid = ex["output"]

                max_y = max(len(input_grid), len(output_grid))

                padded_input = pad_grid(input_grid, max_input_x, max_y, test_output=False)
                padded_output = pad_grid(output_grid, max_output_x, max_y, test_output=True)

                assert len(padded_input) == len(padded_output), \
                    "Padded input and output grids must have the same number of rows."
                assert len(padded_input[0]) == len(padded_output[0]), \
                    "Padded input and output grids must have the same number of columns."

                padded_input_np = np.array(padded_input)
                padded_output_np = np.array(padded_output)

                boundary = np.array([arc_sep_grid_token] * padded_input_np.shape[0])
                padded_example = np.concatenate((padded_input_np, boundary[:, None], padded_output_np), axis=1)

                inpaint_grid.append(padded_example)

            inpaint_grid_flattened = inpaint_grid.flatten().tolist()
            inpaint_grid_str = "".join(inpaint_grid_flattened)
            inpaint_grid_tok = tokenizer.encode(inpaint_grid_str)
            model_inputs["input_ids"].append(inpaint_grid_tok)
            model_inputs["attention_mask"].append([1] * len(inpaint_grid_tok))
            model_inputs["src_mask"].append([1] * (len(inpaint_grid_flattened) + 1))

        return model_inputs

    def print_ar_unsupervised_example(input_ids: List[int], attention_mask: List[int], src_mask: List[int]):
        """Print an example from the ARC dataset."""
        # Decode the input_ids to get the original string
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        # Print the decoded input
        print("Decoded input:\n{}".format(decoded_input))
        # regex match to split the decoded input by a continuous string of repeated <arc_sep_example>
        split_input = re.split(r"(<arc_sep_example>)+", decoded_input)
        # Print the split input
        for i, part in enumerate(split_input):
            print("Example {}:\n{}".format(i, part))

    # if stage != "sft":
    #     # diffusion training & inference
    #     preprocess_func = preprocess_s2s_dataset
    #     print_function = print_unsupervised_dataset_example
    # elif stage == "sft" and not training_args.predict_with_generate:
    #     # AR training
    #     preprocess_func = preprocess_supervised_dataset
    #     print_function = print_supervised_dataset_example
    # else:
    #     # AR inference
    #     preprocess_func = preprocess_unsupervised_dataset
    #     print_function = print_unsupervised_dataset_example

    if stage == "mdm":
        preprocess_func = preprocess_masked_diffusion_dataset
        print_function = print_masked_diffusion_example
    elif stage == "sft":
        preprocess_func = preprocess_ar_supervised_dataset
        print_function = print_ar_supervised_example
    else:
        preprocess_func = preprocess_ar_unsupervised_dataset
        print_function = print_ar_unsupervised_example

    if data_args.cache_path is not None and os.path.exists(data_args.cache_path):
        logger.warning("Loading dataset from disk will ignore other data arguments.")
        return load_from_disk(data_args.cache_path)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        print(f"column_names: {column_names}")
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=column_names,
            **kwargs
        )

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_file`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
