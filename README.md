# Replication Study on BiRT Architecture (Bio-inspired replay for transformers in Continual Learning)

Project Organization
------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    |   |__ViT_CiFAR10_data <- CiFAR10 dataset
    |   |__VIT_CiFAR100_data <- CiFAR100 dataset
    │    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    |   |__models_CiFAR10 <- model_g, model_f_w, model_f_s weights for CiFAR10
    |   |   |__model_g <- model_f_s weights for CiFAR100
    |   |   |__model_f_s <- model_f_s weights for CiFAR100
    |   |   |__model_f_w <- model_f_w weights for CiFAR100 
    |   |__models_CiFAR100  <- model_g, model_f_w, model_f_s weights for CiFAR100
    |   |   |__model_g <- model_f_s weights for CiFAR100
    |   |   |__model_f_s <- model_f_s weights for CiFAR100
    |   |   |__model_f_w <- model_f_w weights for CiFAR100 
    |   |__pretrained_models_CiFAR10 <- pretrained models for each step for CiFAR10
    |   |__pretrained_models_CiFAR100 <- pretrained models for each step for CiFAR100
    |
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── dataset.py           <- Scripts to download or generate data
    |   |__ const.py           <- contains all the constants
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── arch.py     <-  architecture code of BiRT
    │   │   └── train_task.py    <- training code for BiRT for each tasks
    │   │   └── train_finetune_balanced_dataset.py    <- finetuning model on balanced dataset
    |   |   |__ test.py       <- testing code for architecture
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

--------

This repository implements the BiRT architecture for continual learning, as described in the associated paper. The implementation was developed using V100 GPUs with 32GB of GPU RAM and was trained and tested on the CIFAR-10 and CIFAR-100 datasets.

# Environment

- **Python Version:** Python 3.11
- **Libraries:** Listed in `requirements.txt`
- **GCC Version:** `gcc` (GCC) 4.8.5, Red Hat 4.8.5-44 (2015-06-23)
- **GPU:** NVIDIA V100, Node 1

# Running the Implementation

To run this implementation, follow the commands for each task as outlined below. The tasks should be trained sequentially in the order 0, 1, 2, etc. 
Note that you may need to modify the `const.py` file in `src/models` according to the specific experiment you wish to run and the specific task you need to train.
Also make sure `PRETRAINED` is set to `False` in `const.py` before running the test scripts.
Refer to the comments in `const.py` for detailed instructions.

## Training and Testing

```bash
python -m src.models.train_task
python -m src.models.test
python -m src.models.train_finetune_balanced_dataset
python -m src.models.test
```

# Testing with Pretrained Models

All pretrained models are stored in the `models/pretrained_models_$DATASET$` directory in `.pth` format. Make sure `PRETRAINED` is set to `True` in `const.py`. Specify the file path for each `fs`, `fw`, `g` model for the particular task in `const.py`. See const.py for further detailed instructions.

## Model Naming Convention

Each pretrained model follows a specific naming convention to indicate the dataset, task type, task number, and model architecture:

The file naming pattern is as follows: `(dataset)(tasktype)(num)_(modeltype).pth`

### Where:
- **dataset**: Indicates the dataset used, either `cifar10` or `cifar100`.
- **tasktype**: Describes the model’s state:
  - `ft`: Model after fine-tuning.
  - `t`: Model after training for the specific task.
- **num**: Refers to the task number.
- **modeltype**: Specifies the model architecture, which can be:
  - `fs`, `fw`, or `g`.

### Example

For a CIFAR-10 model fine-tuned for task 1 with model architecture `fs`, the file name would be:

`cifar10ft1_fs.pth`


## Testing the Pretrained Models

To test a pretrained model, run the following command:

```bash
python -m src.models.test
```
# Testing the ViT Model

To test the Vision Transformer (ViT) model when decomposed into g() and f() models, execute the Vit_fg.ipynb notebook. To test the original ViT model, run the ViT.ipynb notebook. (Tintn. Vision Transformer from Scratch. https://github.com/tintn/vision-transformer-from-scratch.)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
