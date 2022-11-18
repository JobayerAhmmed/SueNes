# 
# This script is used to test the pre-trained model from bert_tiny_cnndm_tf.py file.
# The pre-trained model is loaded from the saved model.
# The suenes() method takes list of documents and their summaries from user,
# predict the scores of those summaries using the pretrained model,
# and return a list of scores.
# 

from datasets import Dataset
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf
from transformers import AutoTokenizer
from typing import List


def suenes(documents: List[str], summaries: List[str], path_to_model: str) -> List[float]:
    '''
    Predict scores of summaries from the documents and their summaries.

    :param List[str] documents: list of documents
    :param List[str] summaries: list of summaries of the documents
    :param str path_to_model: directory path from where the pre-trained model should be loaded
    :return List[float]: returns the predicted scores of the summaries
    '''
    if len(documents) != len(summaries):
        print('Item count mismatch in documents and summaries')
        return None
        
    model = TFAutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model, model_max_length=512)
    
    dataset = Dataset.from_dict({'text': documents, 'summary': summaries})
    dataset = dataset.map(lambda item: tokenizer(item["text"], item["summary"], padding=True, truncation=True), 
        batched=True)

    tf_dataset = dataset.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        shuffle=False,
        batch_size=8)

    preds = model.predict(tf_dataset)
    result = [round(item[0], 2) for item in preds.logits]
    return result


# 
# Test the model with these sample data
# 
text = "Flying for business or pleasure is one of the most thrilling experiences in the world, but it remains full of mystery for the average traveller. Curious passengers have burning questions every time they step foot on a plane – from whether pilots require a key to start a plane to where flight attendants sleep on long-haul flights. MailOnline Travel spoke to a number of experts to debunk some of the myths that exist and answer some of travellers’ frequently asked questions about planes. Scroll down for video. Once a pilot initiates the sequence to start a modern plane, the remaining steps are done automatically. How do pilots start a plane? It’s not as simple as turning a key or pushing a single button. Starting a plane is ‘a little more complicated’ than starting a car, said Captain Piers Applegarth, a representative of the British Airline Pilots Association (BALPA). The retired training captain said an air start motor rotates the jet engines before adding fuel and starting the ignition. He said: ‘This means that there are a few levers and buttons that need to be moved. On modern aircraft, most of this is done automatically once the start sequence is initiated but a pilot can still start it manually.’ Flight attendants eat the same meals that are provided to passengers if the airline bothers to feed them at all. Is the crew served the same food as passengers? If meals are provided by the airline, pilots do not eat the same reheated chicken or pasta dishes that are served to passengers in economy. To reduce the risk of food poisoning, the captain and co-pilot usually eat different meals, said Mr Applegarth. Like passengers, the flight crew can bring their own food on board, but the meals provided to them are the same or variations of the meals provided to passengers in business class. Caterers will sometimes load meals designated specifically for the crew, said Patrick Smith, a pilot and author of Cockpit Confidential. He said: ‘At my airline we are given a menu prior to departure and will write down our entree preferences – first choice, plus at least one alternate. ‘Eating in the cockpit can be messy, so on international flights I usually take my meals in the cabin, on my rest break. ‘With potential illness in mind, pilots are encouraged to eat different entrees, but this is not a hard and fast rule. In practice it comes to down to your preferences and what’s available.’ Flight attendant Sarah Steegar, who works for a major US airline, said if – ‘and that’s a big if’ – an airline provides food to cabin crew it will be the same food that is served to passengers. But it’s rare for some airlines to provide meals to flight attendants on flights under 12 hours, she added. ‘If there are meals left over we can have that. Many of us try to bring our own food, but it’s a challenge, considering periods of time with no refrigerators and different liquid restrictions and customs laws. ‘Fun fact: the UK is the most difficult when it comes to trying to bring food for ourselves.’ The Airbus A350 has private sleeping quarters for flight attendants on long-haul flights. Where do pilots and cabin crew sleep and go to the loo? Planes that fly long-haul routes which require more than two pilots usually contain private bunks for the flight crew, said Mr Applegarth. In other cases pilots try to catch some shut-eye in special rest seats allocated for them within or near the cockpit, or within the passenger cabin, he added. Mr Applegarth said: ‘Generally flights less than about 10 hours and 30 minutes only carry two pilots. For longer flights extra pilots are carried so that each pilot can have a chance to sleep and be rested for the landing.’ The Airbus A380, which is the world’s largest passenger airliner, has private sleeping quarters below its decks which flight attendants use for rest. Mr Smith said: ‘Flight attendants also work in shifts, and similarly to the pilots their rest quarters can either be a designated block of cabin seats or a separa"
summary1 = "Flying is a thrilling experience but is full of mystery for most passengers. MailOnline Travel spoke to experts to answer common questions. To reduce the risk of food poisoning, pilots do not eat the same meals. Larger planes have private sleeping quarters for flight attendants. ' The world's strongest man' wouldn't be able to open a door mid-flight."
summary2 = "To reduce the risk of food poisoning, pilots do not eat the same meals."
summary3 = "To reduce the risk of food poisoning, pilots do not eat the same meals. The world's strongest man' wouldn't be able to open a door mid-flight."
summaries = [summary1, summary2, summary3]
documents = [text] * 3
path_to_model = "./models/bert_tiny_cnndm_tf"

pred_scores = suenes(documents, summaries, path_to_model)

original_scores = [1.0, 0.2, 0.4]
print('Original scores:', original_scores)
print('Predicted scores:', pred_scores)
