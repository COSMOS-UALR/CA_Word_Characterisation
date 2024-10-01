import re
import json
import yaml 
import os
from unidecode import unidecode
import numpy as np
import importlib.resources as pkg_resources  


def count_conative_ref(conative,referential,text,nlp):
    t = nlp(text)
    return len(conative)/len(t),min(len(referential)/len(t),1)

def count_expressive_emotional(expressive, emotive,text,nlp):
    t = nlp(text)
    sents = list(t.sents)
    return len(expressive)/len(sents),len(emotive)/len(sents)
    
    
def get_token(file_path="token.yml", env_var="API_TOKEN"):
    """
    Function to retrieve the token from environment variables, a YAML file, or prompt the user.

    Parameters:
    - file_path (str): The path to the YAML token file. Defaults to 'token.yml'.
    - env_var (str): The name of the environment variable to fetch the token from. Defaults to 'API_TOKEN'.

    Returns:
    - token (str): The retrieved token.
    """
    
    # 1. Check if the token exists in the environment variables
    token = os.getenv(env_var)
    if token:
        print(f"Token fetched from environment variable '{env_var}'")
        return token

    # 2. Check if the token exists in the YAML file
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            if hasattr(data,'token'):
                token = data.get('token')
            else:
                token=None
            if token:
                print(f"Token fetched from file '{file_path}'")
                return token
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")

    # input the token if not found in environment or file
    if token is None:
        token = input("Enter your Diffbot token: ").strip()
    
    # Optionally, save the token to the YAML file for future use
    save_to_file = input(f"Do you want to save the token to '{file_path}' for future use? (y/n): ").strip().lower()
    if save_to_file == 'y':
        with open(file_path, 'w') as file:
            yaml.dump({'token': token}, file)
        print(f"Token saved to '{file_path}'")
    
    return token

def preprocess_text(text):

    if str(text)=='nan':
        return np.nan


    # Remove emoji
    text = remove_emoji(text)


    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = unidecode(text)

    return text

def remove_emoji(text):
    # Remove emoji using Unicode patterns
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)

def tokenize_with_positions(text):
    tokens = []
    for match in re.finditer(r'\b\w+\b', text):
        token = match.group(0)
        start, end = match.span()
        tokens.append({'token': token, 'start': start, 'end': end})
    return tokens
def are_similar(e1, e2, offset=4):
    range1 = list(range(e1["start"],e1["end"]))
    range2 = list(range(e2["start"],e2["end"]))
    if e1["start"] in range2 or e1["end"] in range2:
        return True
    if e2["start"] in range1 or e2["end"] in range1:
        return True
    return False
def resolve_entities(entities):
    entity_dict = {}
    for entity in entities:
        found = False
        for key, existing_entity in entity_dict.items():
            if are_similar(existing_entity, entity):
                # If a similar token is found, concatenate the labels
                if existing_entity['entity_group']==entity['entity_group']:
                    pass
                else:
                    existing_entity['entity_group'] = f"{existing_entity['entity_group']}_{entity['entity_group']}"
                found = True
                break
        if not found:
            # If no similar token is found, add the entity to the dictionary
            key = (entity['start'], entity['end'])
            entity_dict[key] = entity

    # Convert the dictionary back to a list
    result_entities = list(entity_dict.values())
    return result_entities

def align_entities_with_words(text, entities):
    tokens = tokenize_with_positions(text)
    
    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        
        # Find the closest token start
        for token in tokens:
            if token['start'] <= entity_start < token['end']:
                entity['start'] = token['start']
                break
        
        # Find the closest token end
        for token in tokens:
            if token['start'] < entity_end <= token['end']:
                entity['end'] = token['end']
                break
                
        # Handle cases where entity end is not within any token
        if entity_end > tokens[-1]['end']:
            entity['end'] = tokens[-1]['end']
    
    return entities


non_performing_verbs = set([
    "according", "seem", "appear", "become", "remain", "feel", "look", "sound", "taste", "smell",
    # Stative Verbs
    "be", "know", "have", "believe", "own", "like", "seem", "appear", "contain", 
    "cost", "possess", "belong", "mean", "exist", "need", "lack", "deserve", 
    "owe", "weigh", "fit", "matter", "stand", "remain", "represent", "signify",

    # Perceptual Verbs
    "see", "hear", "smell", "taste", "feel", "notice", "recognize", "perceive", 
    "observe", "detect", "witness", "discern", "spot",

    # Mental Verbs
    "think", "understand", "forget", "imagine", "wish", "realize", "suppose", 
    "assume", "doubt", "consider", "remember", "expect", "know", "guess", 
    "prefer", "hope", "agree", "disagree", "intend", "predict", 
    "acknowledge", "mean", "want", "appreciate", "love", "hate", "fear", 
    "regret", "suspect", "trust", "admire", "envy",

    # Relational Verbs
    "belong", "include", "depend", "consist", "resemble", "match", "equal", 
    "involve", "exclude", "correspond", "vary", "link", "connect", "associate", 
    "relate", "differ", "contrast", "parallel", "coincide", "combine",

    # Speech Act and Emotive Verbs
    "thank", "apologize", "welcome", "congratulate", "blame", "compliment", 
    "accuse", "praise", "commend", "admit", "confess", "declare", "deny", 
    "promise", "request", "suggest", "warn", "argue", "claim", "explain",

    # Additional Non-Performing Verbs
    "concern", "regard", "apply", "accord", "suit", "qualify", "signify", 
    "surround", "adhere", "lack", "result", "stem", "persist", "occur", 
    "remain", "appear", "arise", "fall", "stay", "accord", "pertain", 
    "rest", "hover", "lie", "abide", "exist", "stand", "rest",
    # Primary Auxiliary Verbs
    "be", "am", "is", "are", "was", "were", "being", "been", 

    # Have Auxiliary Verbs
    "have", "has", "had", "having",

    # Do Auxiliary Verbs
    "do", "does", "did", "doing",

    # Modal Auxiliary Verbs
    "can", "could", "will", "would", "shall", "should", 
    "may", "might", "must", "ought to", 

    # Semi-Modal Auxiliary Verbs
    "need", "dare", "used to", "had better"

])





def load_verbs():
    # Ensure that the file is included in the 'data' directory of the package
    with pkg_resources.open_text('ca_model.data','updated_verbs_array.json') as f:
        verbs = json.load(f)['verbs']
    return verbs
extra_conative = load_verbs()
