# Uni-Detect
Implementation of the Uni-Detect paper. 
https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Uni-Detect.pdf

## Requirements
A miniconda environment is provided in the repository. To install it, run:
```
make install
```

## Usage

For preparing the given datalake DGov-141 for testing, run:
```
make create-agg-datalake

make create-datalake-path

make create-datalake-token-dict
```

These commands will create the aggregated datalakes from the given datalake `datasets/DGov-141/separated` for the clean and dirty tables.
The aggregated datalakes are stored in the `datasets/DGov-141/aggregated_dirty` and `datasets/DGov-141/aggregated_clean` directories.
Note: The separated datalake needs to be in the form of `datasets/lake-name/separated/table-name/clean.csv|dirty.csv`.

Inside the output directory `output/DGov-141`, the `output/DGov-141/test_path.pkl` file is created. This file contains the path to the tables inside aggregated datalakes.
Furthermore, the `output/DGov-141/tokens/tokens_dict.pkl` file is created. This file contains the token dictionary for the tokenized datalake.

The Uni-Detect algorithm can solve the following tasks:
1. Functional Dependency Violation Detection
`make fd-test`
2. Numerical Outlier Detection
`make no-test`
3. Spelling Error Detection
`make se-test`
4. Uniqueness Violation Detection
`make uv-test`

Results are stored in the `output/DGov-141` directory and the logs are stored in the `logs` directory.

## Offline training 

Since the Uni-Detect algorithm is based on the machine learning models, it is required to train the models before running the algorithm.
The model is trained on the WDC-5m dataset, which is not included in the repository due to its size.

The WDC-5m also needs the same preprocessing as the DGov-141 dataset. Please refer to the `Usage` section for the preprocessing steps.
Note, that the configs are set for the DGov-141 dataset and need to be changed for the WDC-5m dataset.

The model needs to be trained for the task given above. For example, to train the model for the FD violation detection task, run:
```
make fd-offline
```
The other tasks can be trained in the same way.
```
make no-offline
make se-offline
make uv-offline
```

The trained model is stored in the `output/WDC-5m` directory.

## Uninstall

To uninstall the conda environment, run:
```
make uninstall
```
