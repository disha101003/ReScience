### Replication Study on BiRT Architecture (Bio-inspired replay for transformers in Continual Learning)

Project Organization
------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    |   |__ViT_CiFAR10_data <- CiFAR10 dataset
    |   |__VIT_CiFAR100_data <- CiFAR100 dataset
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    |   |__models_CiFAR10 <- model_g, model_f_w, model_f_s weights for CiFAR10
    |   |   |__model_g <- model_f_s weights for CiFAR100
    |   |   |__model_f_s <- model_f_s weights for CiFAR100
    |   |   |__model_f_w <- model_f_w weights for CiFAR100 
    |   |__models_CiFAR100  <- model_g, model_f_w, model_f_s weights for CiFAR100
    |       |__model_g <- model_f_s weights for CiFAR100
    |       |__model_f_s <- model_f_s weights for CiFAR100
    |       |__model_f_w <- model_f_w weights for CiFAR100 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
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
    │   │   └── train.py    <- training code for BiRT for all tasks
    │   │   └── train_finetune_balanced_dataset.py    <- finetuning model on balanced dataset
    |   |   |__ test.py       <- testing code for architecture
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

This repository implements the BiRT architecture for continual learning, as described in the associated paper. The implementation was developed using V100 GPUs with 32GB of GPU RAM and was trained and tested on the CIFAR-10 and CIFAR-100 datasets.

## Environment

- **Python version:** Python 3
- **Libraries:** Listed in `requirements.txt`

## Running the Implementation

To run this implementation, follow the commands for each task as outlined below. Note that you may need to modify the `const.py` file in `src/models` according to the specific experiment you wish to run. Refer to the comments in `const.py` for detailed instructions.

### Training and Testing

```bash
python -m src.models.train
python -m src.models.test
python -m src.models.train_finetune_balanced_dataset
python -m src.models.test
```
### Testing the ViT Model

To test the Vision Transformer (ViT) model when decomposed into g() and f() models, execute the Vit_fg.ipynb notebook. To test the original ViT model, run the ViT.ipynb notebook.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
