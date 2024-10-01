import requests
import pandas as pd

from typing import List,Dict

FIELDS = "entities,sentiment,facts"
HOST = "nl.diffbot.com"



def get_request(payload,TOKEN):
    

    res = requests.post("https://{}/v1/?fields={}&token={}".format(HOST, FIELDS, TOKEN), json=payload)
    ret = None
    try:
        ret = res.json()
    except:
        print("Bad response: " + res.text)
        print(res.status_code)
        print(res.headers)
    return {'result':ret}

def get_referential_using_spacy(data):

    entities =[]

    for ent in data.doc.ents:
        entities.append({"text":ent.text, "start":ent.start_char, "end":ent.end_char, "entity_group":"referential"})
    return entities

def get_ner(data,use_spacy=False)->List[Dict]:
    entities = []
    try:
        if use_spacy:
            entities = get_referential_using_spacy(data)
            return entities
    except Exception as e:
        print("unable to use spacy to extract NER",{e})

    try:
        res = get_request({
        "content": data.text,
        "lang": "en",
        "format": "plain text with title",
        },data.token)
        res = res['result']
        if res:
            if "entities" in res:
                df = pd.DataFrame.from_dict(sorted(res["entities"], key = lambda ent: ent["salience"], reverse=True))
                for i in df.index:
                    mentions = df.mentions[i]
                    for mention in mentions:
                        if not 'isPronoun' in mention:
                            entities.append({"text":mention['text'],"start":mention['beginOffset'],"end":mention["endOffset"],"entity_group":"referential"})
        else:
            print("unable to use diffbot to extract NER")
            print(" using spacy model to extract referntial element",flush=True)
    except Exception as e:
        print("unable to use diffbot to extract NER",{e})
        print(" using spacy model to extract referntial element",flush=True)
        entities = get_referential_using_spacy(data)
    return entities

