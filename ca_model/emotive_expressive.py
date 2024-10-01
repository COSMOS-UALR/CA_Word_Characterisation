from transformers import pipeline
from typing import List,Dict
expressive_category  =['approval',"disapproval","disappointment","confusion","realization","admiration","optimism","desire","grief"]

def extract_expressive_emotive(data,emotion_classifier)->List[Dict]:

    emotive =[]
    expressive =[]
    for sent in data.doc.sents:
        emotion = emotion_classifier(sent.text)
        if emotion[0]["label"] in expressive_category:
            expressive.append({"text":sent.text,"start":sent.start_char,"end":sent.end_char,"entity_group":"expressive"})
        elif emotion[0]["label"]!="neutral":
            emotive.append({"text":sent.text,"start":sent.start_char,"end":sent.end_char,"entity_group":"emotive"})
            
            
    return emotive,expressive
