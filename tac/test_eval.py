import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
import os, json, csv
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=4)

article_per_set = 10
num_summarizer = 47

def read_tac_test_result(BERT_result_file, tac_json, human_only=True):
    """Load BERT result of using TAC2010 as test set and average over 10 articles

    By BERT convention, the file name is test_result.tsv 
    46 document sets, 47 summarizers (2 baselines, 41 machine ones, and 4 humans)

    one number per line 

    Due to how samples are generated, this is the correspondence between lines and doc-sum pairs
    docset_1: 
        article_1:
            summarizer_1, prediction 
            summarizer_2, prediction 
            ...
            summarizer_47, prediction 
        article_2: 
            summarizer_1, prediction 
            summarizer_2, prediction 
            ...
            summarizer_47, prediction 
        ...
        article_10: 
            ...
    docset_2: 
    ...
    docset_46: 

    what we want is:
    docset 1:
        summarizer_1, average of 10 predictions for article_{1,2,...,10}
        summarizer_2, average of 10 predictions for article_{1,2,...,10}
        ...
        summarizer_47, average of 10 predictions for article_{1,2,...,10}
    docset 2:
    ...
    docset_46: ...
    """

    with open(BERT_result_file, "r") as f:
        all_lines = f.read() 

    lines = all_lines.split("\n")
    if lines[-1] == "":
        lines = lines[:-1]  # get rid of last empty line 

    tac = json.load(open(tac_json, 'r'))
    score_dict = {}  # keys as (docset, summarizer), values as list of 10 floats
    docset_counter, article_counter, summarizer_counter = 0,0,0
    # Note that docset_counter, article_counter, nor summarizer_counters is actual docset, article or summarizer IDs. It's just counters to know whether we loop to next article. 
    
    for line in lines:
            # print (docset_counter, article_counter, summarizer_counter)
        docset = list(tac.keys())[docset_counter]
        summarizer = list(tac[docset]["summaries"].keys())[summarizer_counter]
        key = (docset, summarizer)
        if (human_only and summarizer.isnumeric()) or not human_only :
            score_dict.setdefault(key, []).append(float(line))
        
        if summarizer_counter == 47 - 1:
            summarizer_counter = 0
            if article_counter == 10 - 1: 
                article_counter = 0
                docset_counter += 1 
            else:
                article_counter += 1 
        else:
            summarizer_counter += 1 


    # Now, convert to the order in tac and get average 
    score_sorted = [] 
    for docset in tac.keys():
        for summarizer in tac[docset]["summaries"].keys():
            if (human_only and summarizer.isnumeric()) or not human_only :
                ten_scores = score_dict[(docset, summarizer)]
                avg_score = sum(ten_scores)/len(ten_scores)
                score_sorted.append(avg_score)

    return score_sorted

def load_tac_json(task_json, human_only=True):
    """Load the human scores from TAC from the JSON file compiled and dumped by our tac.py script 

    task_json: the JSON file containing all TAC samples and their human scores
    result_file: the test_result.txt by BERT on the test set

    The order of extracting scores from task_json needs to match that in _pop_tac_samples() in run_classifier.py


    order:
    docset_1, summarizer_1, scores[0:2]
    docset_1, summarizer_2, scores[0:2]
    ...
    docset_1, summarizer_47, scores[0:2]

    docset_2, summarizer_1, scores[0:2]
    docset_2, summarizer_2, scores[0:2]
    ...
    docset_2, summarizer_47, scores[0:2]
    ...
    ...
    ...
    docset_46 

    return: 
      dict: (docset_ID, summarizerID): scores[0:2]

    """

    tac_scores = [] # 46 x 47 rows, 3 columns

    tac = json.load(open(task_json, 'r'))
    for docset in tac.keys():
        if human_only: 
            tac_scores += [ tac[docset]["summaries"][summarizer]["scores"] for summarizer in tac[docset]["summaries"].keys() if summarizer.isnumeric() ]
        else:
            tac_scores += [ tac[docset]["summaries"][summarizer]["scores"] for summarizer in tac[docset]["summaries"].keys()]

    return tac_scores 

def calc_cc(tac_results, tac_scores):
    """Compute the correlation coefficients between BERT results on TAC test set and human evaluated scores on TAC test set

    tac_results: 1-D list of floats, 46(docset)x47(summarizers) elements
    tac_scores: 2-D list of floats, 46(docset)x47(summarizers) rows, and 3 columns
    """
    tac_scores = np.array(tac_scores)

    corr_pearsons = [pearsonr(tac_results, tac_scores[:, i])[0] for i in range(3)]
    corr_spearmans = [spearmanr(tac_results, tac_scores[:, i])[0] for i in range(3)]

    line = corr_pearsons + corr_spearmans
    line = ["%.5f"%i for i in line]
    line = "\t".join(line)

    print (line)

    # for i in range(3):
    #     corr_pearson = pearsonr(tac_results, tac_scores[:, i])
    #     corr_spearman = spearmanr(tac_results, tac_scores[:, i])

    #     print(corr_pearson[0], corr_spearman[0])
    # print("---------------------------")


def cc_all(plot = True):
    BERT_result_prefix = "../bert/result_base/scientific_papers"
    tac_json_file = "../bert/TAC2010_all.json"
    human_only=True

    for method in ["sent_delete"]:
        print (method)
        BERT_result_file = os.path.join(BERT_result_prefix, method, "test_results.tsv")
        tac_results = read_tac_test_result(BERT_result_file, tac_json_file, human_only)
        tac_scores = load_tac_json(tac_json_file, human_only)
        calc_cc(tac_results, tac_scores)
        
        if plot:
            tac_scores = np.array(tac_scores)
            plt.figure(figsize=(12, 3))
            ax = plt.subplot(1, 3, 1)
            sns.kdeplot(tac_scores[ :, 0], label='Modified')
            sns.kdeplot(tac_results, label='Prediction')
            ax.legend()
            ax = plt.subplot(1, 3, 2)
            sns.histplot((tac_scores[ :, 1]-1)/4, stat='density', bins=5, label='Linguistic')
            sns.kdeplot(tac_results, label='Prediction', color='tab:orange')
            ax.legend()
            ax = plt.subplot(1, 3, 3)
            sns.histplot((tac_scores[ :, 2]-1)/4, stat='density', bins=5, label='Overall')
            sns.kdeplot(tac_results, label='Prediction', color='tab:orange')
            ax.legend()
            plt.show()
        
if __name__ == "__main__":
    cc_all()
