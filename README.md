# SueNes: A Weakly Supervised Approach to Evaluating Single-Document Summarization via Negative Sampling

[Here is NAACL 2022 paper](https://aclanthology.org/2022.naacl-main.175/)

## Repeat Transformer-based Experiments
<!-- Please read [README.md](transformer/README.md) -->
The `transformer` directory contains code for training transformer-based models with different datasets.
The datasets were generated using `sentence delete` or `word delete` techniques
mentioned in the SueNes [paper](https://aclanthology.org/2022.naacl-main.175/).

### Environmet Setup
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

### Generate Datasets
- `mkdir exp exp/data exp/result`
- `cd pre`
- `python3 sentence_scramble.py`
- `python3 sample_generation.py`

### Bert Tiny CNN Daily Mail TensorFlow
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

### Bert Tiny CNN Daily Mail PyTorch
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


<!-- ## Dependencies and environment
* The negative sampling code requires TF 2.x and [`tensorflow_datasets`](https://www.tensorflow.org/datasets). 
* The `bert` code requires TF 1.15. We run our experiments using [nVidia's TF fork](https://github.com/NVIDIA/tensorflow). 
* SpaCy is needed for segmentation. 
* System: Ubuntu 20.04, 64GB RAM, RTX 3090


## To repeat our experiments

First, create folders under the directory of this project:

```bash
mkdir exp exp/data exp/result
```

### 1. Negative sampling
Code for generating negative samples are in `pre` folder. 

```bash
cd pre
python3 sentence_scramble.py # for sentence-level mutations 
python3 sample_generation.py # for crosspairing and word-level mutations
```
Configrations corresponding to the two Python scripts above are in  `sentence_conf.py` and `sample_generation.py`. Edit them to change negative sampling settings. 

### 2. Model training and test 

Code for model training and test is in the `bert` folder. 

Suppose now you are still in `pre` folder. 
```bash
cd ../bert # go one level up and then into the bert folder 
bash run_classifier.sh 
```

It will call our modified BERT's `run_classifier.py` script to train negative samples just generated above and to test on Newsroom, RealSumm, and TAC2010. Variable names in our `run_classifier.sh` bash script are made very self-explaintory for you to conveniently change the settings, such as the training set, test set, etc. 

Our `run_classifier.py` script hard-codes paths for the three test sets as: `./newsroom_60.tsv`, `./realsumm_100.tsv`, and `./TAC2010_all`. The files `newsroom_60.tsv` and `realsumm_100.tsv` are in this repo for convenience. TAC2010 is not because its access requires approval from NIST. All three files can be generated from raw data using scripts under `human` folder. Please refer to the README file under `human/{newsroom, realsumm, tac}` for information. 

### 3. Aligning with human evaluations 

Code for computing the correlation between our models' predictions and human ratings from the three datasets is in the `human` folder. 

## MISC 
Additional code are kept for reference, e.g., used in early stage of the development of our appproach:   
* `embed`: Scripts for sentence-level embedding. Kept for reference.
* `old`: Sentence-level models. Kept for reference. 

## Baselines and upperbounds

### Baselines: without using human-written reference summaries
* SUPERT: using heuristics to generate psedo-summaries
* BLANC: converting summary quality assessment into a question anwersing problem
* SummaQA: converting summary quality assessment into a question anwersing problem
* SUM-QE and WS_Score: problematic work 

### Upperbounds: using human-written reference summaries -->
