import os
from typing import TYPE_CHECKING, List, Union
from datasets import concatenate_datasets, interleave_datasets, load_dataset

from llmtuner.dsets.utils import checksum, EXT2TYPE
from llmtuner.extras.logging import get_logger
from datasets import Dataset, IterableDataset

if TYPE_CHECKING:
    from llmtuner.hparams import ModelArguments, DataArguments

logger = get_logger(__name__)


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    max_samples = data_args.max_samples
    all_datasets: List[Union["Dataset", "IterableDataset"]] = []  # support multiple datasets

    for dataset_attr in data_args.dataset_list:
        logger.info("Loading dataset {}...".format(dataset_attr))

        # if dataset_attr.load_from == "hf_hub":
        #     data_path = dataset_attr.dataset_name
        #     data_files = None
        # elif dataset_attr.load_from == "script":
        #     data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        #     data_files = None
        # elif dataset_attr.load_from == "file":

        if dataset_attr.load_from == "file":
            # data_path = None
            # data_files: List[str] = []

            dataset_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_path)
            data_type = None
            data_files: List[str] = []

            if os.path.isdir(dataset_path):  # directory
                for file_name in os.listdir(dataset_path):
                    # data_files.append(os.path.join(dataset_path, file_name))
                    data_files.append(file_name)
                    print(f"data_files: {data_files}")
                    # ensure all files have the same type (csv, json, jsonl, txt)
                    if data_type is None:
                        data_type = EXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_type == EXT2TYPE.get(file_name.split(".")[-1], None), "file type does not match."
            elif os.path.isfile(dataset_path):  # single file
                data_files.append(dataset_path)
                data_type = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
            else:
                raise ValueError("File(s) not found.")

            assert data_type, "File extension must be txt, csv, json or jsonl."
            # checksum(data_files, dataset_attr.dataset_sha1)
        else:
            raise NotImplementedError

        dataset = load_dataset(
            dataset_path,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
            # use_auth_token=True if model_args.use_auth_token else None
        )

        # dataset = Dataset.from_dict(dataset[:1000])

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        # TODO: adapt to the sharegpt format

        # for column_name in ["prompt", "query", "response", "history"]:  # align datasets
        #     if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
        #         dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

        # if dataset_attr.system_prompt:  # add system prompt
        #     system_prompt = dataset_attr.system_prompt
        #     if data_args.streaming:
        #         dataset = dataset.map(lambda _: {"system": system_prompt})
        #     else:
        #         dataset = dataset.add_column("system", [system_prompt] * len(dataset))

        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        return all_datasets[0]
    # elif data_args.mix_strategy == "concat":
    #     if data_args.streaming:
    #         logger.warning("The samples between different datasets will not be mixed in streaming mode.")
    #     return concatenate_datasets(all_datasets)
    # elif data_args.mix_strategy.startswith("interleave"):
    #     if not data_args.streaming:
    #         logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
    #     return interleave_datasets(
    #         datasets=all_datasets,
    #         probabilities=data_args.interleave_probs,
    #         seed=data_args.seed,
    #         stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
    #     )
    # else:
    #     raise ValueError("Unknown mixing strategy.")
