from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
tokenizer = AutoTokenizer.from_pretrained(model)

inputs = tokenizer("I loved reading the Hunger Games!")
inputs


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = tokenizer.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# text = "Here is the sentence I want embeddings for."
# marked_text = "[CLS] " + text + " [SEP]"

# # Tokenize our sentence with the BERT tokenizer.
# tokenized_text = tokenizer.tokenize(marked_text)

# # Print out the tokens.
# print (tokenized_text)