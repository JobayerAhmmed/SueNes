The `transformer` directory contains code for training transformer-based models with different datasets.
The datasets were generated using `sentence delete` or `word delete` techniques
mentioned in the SueNes [paper](https://aclanthology.org/2022.naacl-main.175/).

## Environmet Setup
You can create virtual environment using Python or Conda.

#### Python venv (CPU Only)
- `git clone https://github.com/JobayerAhmmed/SueNes.git`
- `cd SueNes`
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python -m spacy download en_core_web_sm`
- `pip install transformers datasets scikit-learn evaluate pyyaml h5py`
- Issue: replace `from keras.saving.hdf5_format` by `from tensorflow.python.keras.saving.hdf5_format` 
    at line 39 of `.venv/lib/python3.10/site-packages/transformers/modeling_tf_utils.py`

#### Conda venv (GPU)
- Create venv following [this](https://www.tensorflow.org/install/pip#linux) documentation
- `pip install tensorflow tensorflow-datasets tensorflow_hub`
- Install PyTorch following [this](https://pytorch.org/get-started/locally/) documentation
- `pip install joblib numpy nltk matplotlib bs4 spacy stanza`
- `python -m spacy download en_core_web_sm`
- `pip install transformers datasets scikit-learn evaluate pyyaml h5py`

## Generate Datasets
- `mkdir exp exp/data exp/result`
- `cd pre`
- `python3 sentence_scramble.py`
- `python3 sample_generation.py`

## Bert Tiny CNN Daily Mail TensorFlow
Code for the model is in `bert_tiny_cnndm_tf.py` file.
This model is trained from checkpoint found in 
[prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny). 
Data is generated from CNN Daily Mail dataset using 
[SueNes](https://github.com/forrestbao/SueNes).
Only `sentence delete` technique, defined in 
SueNes [paper](https://aclanthology.org/2022.naacl-main.175/),
is used for data generation.
Only 10% data is considered from CNN Daily Mail dataset's train split
for generating train split for our experiment.

#### Train Model
- `cd transformer`
- `python3 bert_tiny_cnndm_tf.py`

#### Test Model
- `python3 bert_tiny_cnndm_tf_wrap.py`

## Bert Tiny CNN Daily Mail PyTorch
Code for the model is in `bert_tiny_cnndm_pt.py` file.
This model is trained from checkpoint found in 
[prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny). 
Data is generated from CNN Daily Mail dataset using 
[SueNes](https://github.com/forrestbao/SueNes).
Only `sentence delete` technique, defined in 
SueNes [paper](https://aclanthology.org/2022.naacl-main.175/),
is used for data generation.
Only 10% data is considered from CNN Daily Mail dataset's train split
for generating train split for our experiment.

#### Train Model
- `cd transformer`
- `python3 bert_tiny_cnndm_pt.py`

#### Test Model
- `python3 bert_tiny_cnndm_pt_wrap.py`

