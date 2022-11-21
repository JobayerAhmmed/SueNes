#
# This model is a TensorFlow pre-trained model from the checkpoint: https://huggingface.co/prajjwal1/bert-tiny
# The training data is generated from CNN Daily Mail dataset.
# Data generation is done by SueNes: https://github.com/forrestbao/SueNes
# From CNN Daily Mail dataset, we considered 10% for train, 2.5% for validation and 2.5% for test data.
# Please read README.md file to learn about data generation.
# 

import os
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification


def process_data(split: str):
    '''Read tsv file and return panda dataframes.'''
    data_dir = '../exp/data/cnn_dailymail/sent_delete_sent/'
    filepath = data_dir + split + '.tsv'
    data = []
    with open(filepath) as file:
        for item in file:
            items = item.split('\t')
            text = items[0]
            for i in range(2, len(items), 2):
                data.append([items[0], items[i-1], float(items[i])])
    cols = ['text', 'summary', 'score']
    df = pd.DataFrame(data, columns=cols)
    return df


def prepare_dataset(split: str):
    data = process_data(split)
    return Dataset.from_pandas(data)


def bert_tiny_cnndm_tf():
    train_dataset = prepare_dataset('train')
    validation_dataset = prepare_dataset('validation')

    # Tokenize data
    model_checkpoint = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)

    train_dataset = train_dataset.map(lambda item: tokenizer(item["text"], item["summary"], padding=True, truncation=True), 
        batched=True)
    validation_dataset = validation_dataset.map(lambda item: tokenizer(item["text"], item["summary"], padding=True, truncation=True), 
        batched=True)

    tf_train_dataset = train_dataset.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"], label_cols=["score"],
        shuffle=True, batch_size=8)
    tf_validation_dataset = validation_dataset.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"], label_cols=["score"],
        shuffle=False, batch_size=8)

    # Model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1, from_pt=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.MeanSquaredError())

    # This part is optional.
    # You can save your model while training.
    # If your process crashes while training, you can 
    # retstart the training from just before the crash.
    models_dir = "./models/bert_tiny_cnndm_tf"
    checkpoint_path = models_dir + "/checkpoints/cp-{epoch}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    callbacks = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq="epoch")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    # Train the Model
    model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=callbacks)

    # Save the Trained Model
    # 
    # Other people will be able to load your trained model
    # and use for predicting the result
    tokenizer.save_pretrained(models_dir)
    model.save_pretrained(models_dir)

    # Predict score for validation dataset (optional)
    # 
    # If you pass a dataset, it will predict the result for all data of the dataset.
    model.predict(tf_validation_dataset)


if __name__ == "__main__":
    bert_tiny_cnndm_tf()