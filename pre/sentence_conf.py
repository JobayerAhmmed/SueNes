# configuration file for senentece-level negative sampling 

import string

# ==== constants about datasets 
dataset_features = {"cnn_dailymail": ['article', 'highlights'],
    "big_patent": ['description', 'abstract'],
    "newsroom": ['text', 'summary'],
    "scientific_papers": ['article', 'abstract'],
    "billsum":['text','summary'],
    "dryrun":[]
    }

dataset_sizes = {
    "cnn_dailymail":311971, 
    "big_patent":1341362
}

dataset_sizes_w_split = {# new for sentence-level mutation
    "cnn_dailymail":{'train':28711, 'test':1149, 'validation': 1336}, # 10%
    "big_patent":{'train':120722, 'test':6707, 'validation': 6706}, # 10%
}

#======== data loading parameters 

# Must match their names in TFDS 
# dataset_name = "dryrun" 
dataset_names = ["cnn_dailymail", "big_patent"] 

splits = ['train', 'test', 'validation'] 

#========= data output/dumping parameters 

data_root = "../exp/data"  # new for sentence-level mutation
data_dir = "/data/NLP"  # comment this line, or change it according to your local PC

n_jobs = 35

# compact or plain 
# plain is 3-column, doc, summary, target
# but plain may contain repeated docs, 
# which will cause extra time in sentence embedding (not applicable for BERT) 
# compact: small. easy for inspecting dump. Format per line: 
# doc, sum1, label1, sum2, label2, sum3, label3, ...

dump_format = "compact"

my_batch_size = 2**8*64
# how many doc-sum pairs to process each time
# When using stanza, too large or too small reduces GPU utility rate. 2**8 is a good number.
# The speed is about 10 seconds per 2**8 doc-sum pairs on 3090
# Doesn't matter when using Spacy. Set it to 2**8*64 on CPU. Adjust based on your RAM.

#========= NLP parameters

special_characters_to_clean = ['\n', '\t'] # replace such strings in raw data 

sent_end = [".", "!", "?"]  # symbols that represent the end of a sentence 
sent_end = string.punctuation

tokenizer_name = 'spacy' # or "stanza", "nltk" 
spacy_batch_size = 8000 # doesn't seem to have much effect though

#========= negative sampling parameters 

# ratio between negative and positive samples
# minimal: 1 
neg_pos_ratio = 5

# methods used to generate negative samples 
methods = ["sent_delete"] 
# mode = 'char' # or 'sent'  # measure how many sentences or characters are altered. # TODO: add token one
mode = 'sent'
