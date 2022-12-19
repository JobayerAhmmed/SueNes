#!/bin/sh
echo "Creating and activating virtual environment ..."
python3 -m venv .venv
source .venv/bin/activate
echo "Done activation virtual environment"
echo "=============================================================================================="
echo "Executing pip install -r requirements.txt"
pip install -r requirements.txt
echo "Done pip install -r requirements.txt"
echo "=============================================================================================="
echo "Executing python3 -m spacy download en_core_web_sm"
python3 -m spacy download en_core_web_sm
echo "Done python3 -m spacy download en_core_web_sm"
echo "=============================================================================================="
echo "Executing pip install transformers datasets scikit-learn evaluate pyyaml h5py"
pip install transformers datasets scikit-learn evaluate pyyaml h5py
echo "Done pip install transformers datasets scikit-learn evaluate pyyaml h5py"
echo "=============================================================================================="
echo "Executing mkdir exp exp/data exp/result"
mkdir exp exp/data exp/result
echo "Done mkdir exp exp/data exp/result"
echo "=============================================================================================="
echo "Executing python3 pre/sentence_scramble.py"
python3 pre/sentence_scramble.py
echo "Done python3 pre/sentence_scramble.py"
echo "=============================================================================================="
echo "Executing python3 transformer/bert_tiny_cnndm_tf.py"
python3 transformer/bert_tiny_cnndm_tf.py
echo "Done python3 transformer/bert_tiny_cnndm_tf.py"
echo "=============================================================================================="
echo "Executing python3 transformer/bert_tiny_cnndm_tf_wrap.py"
python3 transformer/bert_tiny_cnndm_tf_wrap.py
echo "Done python3 transformer/bert_tiny_cnndm_tf_wrap.py"
echo "=============================================================================================="
echo "Executing python3 transformer/bert_tiny_cnndm_pt.py"
python3 transformer/bert_tiny_cnndm_pt.py
echo "Done python3 transformer/bert_tiny_cnndm_pt.py"
echo "=============================================================================================="
echo "Executing python3 transformer/bert_tiny_cnndm_pt_wrap.py"
python3 transformer/bert_tiny_cnndm_pt_wrap.py
echo "Done python3 transformer/bert_tiny_cnndm_pt_wrap.py"
echo "=============================================================================================="
deactivate