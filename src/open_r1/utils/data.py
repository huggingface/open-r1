from typing import Dict, List, Optional, Union
import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
import logging
import numpy as np

from ..configs import DatasetConfig, ScriptArguments

logger = logging.getLogger(__name__)

def get_dataset(
    args: ScriptArguments,
) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset or a mixture of datasets based on the configuration.
    
    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.
    
    Returns:
        Dataset or DatasetDict: The loaded and processed dataset(s).
    """
    # Case 1: Using a single dataset
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        return datasets.load_dataset(args.dataset_name, args.dataset_config_name)
        
    # Case 2: Using a dataset mixture
    elif args.dataset_mixture:
        logger.info(f"Loading dataset mixture with {len(args.dataset_mixture)} datasets")
        
        # Dictionary to store datasets by split
        mixture_by_split = {}
        
        # Process each dataset in the mixture
        for dataset_name, dataset_config in args.dataset_mixture.items():
            logger.info(f"Loading dataset for mixture: {dataset_name}")
            
            # Load the dataset
            ds = datasets.load_dataset(
                dataset_name,
                dataset_config.config
            )
            dataset_split = dataset_config.split

            if dataset_split in ds:
                split_ds = ds[dataset_split]

                # Filter columns if specified
                if dataset_config.columns:
                    split_ds = split_ds.select_columns(dataset_config.columns)

                # Apply dataset weight as a sampling fraction
                if dataset_config.weight < 1.0:
                    # Calculate how many samples to keep
                    n_samples = int(len(split_ds) * dataset_config.weight)
                    if n_samples <= 0:
                        logger.warning(
                            f"Weight {dataset_config.weight} for dataset {dataset_name} "
                            f"results in 0 samples. Using 1 sample instead."
                        )
                        n_samples = 1
                    
                    # Randomly sample the dataset
                    indices = np.random.choice(
                        len(split_ds), 
                        size=n_samples, 
                        replace=False
                    )
                    split_ds = split_ds.select(indices)
                    
                    logger.info(
                        f"Applied weight {dataset_config.weight} to dataset {dataset_name}: "
                        f"selected {n_samples} examples out of {len(ds[dataset_split])}"
                    )
                
                # Add to the appropriate split in our mixture
                if dataset_split not in mixture_by_split:
                    mixture_by_split[dataset_split] = []
                
                mixture_by_split[dataset_split].append(split_ds)
            else:
                logger.warning(f"Split {dataset_split} not found in dataset {dataset_name}")
        
        # Concatenate datasets for each split
        for split_name, datasets_list in mixture_by_split.items():
            if datasets_list:
                mixture_by_split[split_name] = concatenate_datasets(datasets_list)
                logger.info(f"Created mixture for split '{split_name}' with {len(mixture_by_split[split_name])} examples")
            else:
                logger.warning(f"No datasets found for split {split_name}")
        
        return DatasetDict(mixture_by_split)
    
    else:
        raise ValueError("Either dataset_name or dataset_mixture must be provided")