#
# This model is a PyTorch pre-trained model from the checkpoint: https://huggingface.co/prajjwal1/bert-tiny
# The training data is generated from CNN Daily Mail dataset.
# Data generation is done by SueNes: https://github.com/forrestbao/SueNes
# From CNN Daily Mail dataset, we considered 10% for train, 2.5% for validation and 2.5% for test data.
# Please read README.md file to learn about data generation.
# 

import pandas as pd
import tensorflow as tf
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


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


def bert_tiny_cnndm_pt():
    train_dataset = prepare_dataset('train')
    validation_dataset = prepare_dataset('validation')

    # Tokenize data
    model_checkpoint = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)

    train_dataset = train_dataset.map(lambda item: tokenizer(item["text"], item["summary"], padding=True, truncation=True), 
        batched=True)
    validation_dataset = validation_dataset.map(lambda item: tokenizer(item["text"], item["summary"], padding=True, truncation=True), 
        batched=True)

    train_dataset = train_dataset.rename_column("score", "labels")
    validation_dataset = validation_dataset.rename_column("score", "labels")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)

    # Metric
    metric = evaluate.load("mse")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training
    models_dir = "./models/bert_tiny_cnndm_pt"
    checkpoint_dir = models_dir + "/checkpoints"

    training_args = TrainingArguments(
        output_dir=checkpoint_dir, 
        evaluation_strategy="epoch", 
        num_train_epochs=3,
        learning_rate=1e-5,
        save_strategy="epoch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    # Train the Model
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    trainer.save_model(models_dir)

    trainer.evaluate()
    

if __name__ == "__main__":
    bert_tiny_cnndm_pt()
