#!/bin/sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
pip install transformers datasets scikit-learn evaluate pyyaml h5py

mkdir exp exp/data exp/result

cd pre
python3 sentence_scramble.py

cd ..
cd transformer

python3 bert_tiny_cnndm_tf.py
python3 bert_tiny_cnndm_tf_wrap.py

python3 bert_tiny_cnndm_pt.py
python3 bert_tiny_cnndm_pt_wrap.py

deactivate