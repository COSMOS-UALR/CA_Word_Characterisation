from spacy import displacy
import pandas as pd
import numpy as np
import pdfkit
from transformers import pipeline
import spacy
import nltk
from .conative import get_conative
from .referential import get_ner
from .emotive_expressive import extract_expressive_emotive
from .utils import get_token,align_entities_with_words,resolve_entities,count_conative_ref,count_expressive_emotional
from tqdm import tqdm
import concurrent.futures
import threading
import os





nlp,emotion_classifier, TOKEN = None, None,None
model_init_lock = threading.Lock()

class Data:
    def __init__(self,text,token,nlp):
        self.text = text
        self.token = token
        self.nlp = nlp
        self.doc = nlp(text)




def download_and_load_model_files():
    # Check if NLTK models are already downloaded
    nltk_data_path = os.path.expanduser('~') + '/nltk_data/tokenizers/punkt'
    if not os.path.exists(nltk_data_path):
        print("Downloading NLTK 'punkt' model")
        nltk.download('punkt')

    
    nltk_pos_path = os.path.expanduser('~') + '/nltk_data/taggers/averaged_perceptron_tagger'
    if not os.path.exists(nltk_pos_path):
        print("Downloading NLTK 'averaged_perceptron_tagger' model")
        nltk.download('averaged_perceptron_tagger')


    # Check if spaCy model is already downloaded
    try:
        nlp = spacy.load("en_core_web_sm")
        
    except OSError:
        print("Downloading spaCy 'en_core_web_sm' model")
        spacy.cli.download("en_core_web_sm")




def initialize_model():
    global nlp, emotion_classifier, TOKEN
    with model_init_lock:  # Ensure that model initialization happens in a thread-safe way
        download_and_load_model_files()
        if nlp is None or emotion_classifier is None or TOKEN is None:
            print("Loading models...")
            # Loading the spacy model and emotion classifier
            nlp = spacy.load("en_core_web_sm")
            emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", max_length=512, truncation=True)
            
            TOKEN = get_token()
            print("Models loaded")
        
            # print("Models are already initialized.")

def extract_entities_from_text(text, do_emotion=False):
    global nlp, emotion_classifier, TOKEN
    # Initialize models if not done yet, using thread-safe initialization
    if nlp is None or emotion_classifier is None or TOKEN is None:
        initialize_model()
    
    # Handle case where the token is invalid
    if TOKEN in [None, '', ' ', '  ']:
        print("No valid token was entered")
        print("Using Spacy NER model")
        TOKEN = None

    # Create data object
    
    data = Data(text, TOKEN, nlp)

    # Emotion analysis (optional)
    emotive = []
    expressive = []
    if do_emotion:
        emotive, expressive = extract_expressive_emotive(data, emotion_classifier)

    # Perform NER
    if data.token is None:
        referential = get_ner(data, use_spacy=True)
    else:
        referential = get_ner(data, use_spacy=False)

    # Conative analysis
    conative = get_conative(data)

    return conative, referential, emotive, expressive






# Function to process texts in parallel
def process_texts_in_parallel(texts, do_emotion=True, max_workers=4):
    # Create a list to store the results in the order of the input texts
    results = [None] * len(texts)
    def align_and_resolve_entities(entities,text):
        try:
            entities = align_entities_with_words(text, entities)

            entities = resolve_entities(entities)
        except Exception as e:
                print(f"Error aligning text: {text}\nError: {e} for entities {entities}")
                
        return entities
    # Helper function to process a single text
    def process_single_text(index, text):
        entities_list = extract_entities_from_text(text, do_emotion)
        result = [align_and_resolve_entities(entities,text) for entities in entities_list]
        
        return index, result

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_index = {
            executor.submit(process_single_text, index, text): index
            for index, text in enumerate(texts)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                index, result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing text at index {index}: {e}")
                results[index] = None  

    return results


class CA_Model:
    def __init__(self) -> None:
        pass

    def get_result_count(self,texts,return_counts=True,max_workers=8):
        self.result_count =[]
        results = process_texts_in_parallel(texts,do_emotion=True,max_workers=max_workers)
        for text,result in zip(texts,results):
            self.result_count.append([count_conative_ref(result[0],result[1],text,nlp),*count_expressive_emotional(result[2],result[3],text,nlp)])
        if return_counts:
            return results,self.result_count
        return results
    
    def get_5_point_statistics(self,result_count=None):

        if not result_count and not self.result_counts:
            raise Exception("counts of each category is needed for 5 point stats.")

        result_count = result_count if result_count else self.result_count
        # Convert the list to a numpy array
        values = np.array(result_count)
        
        # Calculate the five-point summary
        minimum = np.min(values,axis=0)
        q1 = np.percentile(values, 25,axis=0)
        median = np.median(values,axis=0)
        q3 = np.percentile(values, 75,axis=0)
        maximum = np.max(values,axis=0)
        
        # Return the summary as a dictionary or list
        return {
            'Minimum': minimum,
            'Q1 (25th percentile)': q1,
            'Median (50th percentile)': median,
            'Q3 (75th percentile)': q3,
            'Maximum': maximum
    }


    def filter_data_by_date(self, df, date_column, start_date, end_date,timezone='utc'):
        """
        Filters a DataFrame to include rows between the specified start date and end date.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        date_column (str): The name of the column with date information.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format (exclusive).

        Returns:
        pd.DataFrame: A filtered DataFrame containing rows between the specified dates.
        """
        # Convert the date column to datetime if it's not already
        df[date_column] = pd.to_datetime(df[date_column])

        # Convert start and end dates to datetime
        start_date = pd.to_datetime(start_date).tz_localize(timezone)
        end_date = pd.to_datetime(end_date).tz_localize(timezone)
        
        # Filter the DataFrame
        filtered_df = df[(df[date_column] >= start_date) & (df[date_column] < end_date)]
        
        return filtered_df

    def set_token(self, token):
        global TOKEN

        TOKEN = token

def extract_entities_from_text(text, do_emotion=False):
    global nlp, emotion_classifier, TOKEN
    # Initialize models if not done yet, using thread-safe initialization
    if nlp is None or emotion_classifier is None or TOKEN is None:
        initialize_model()
    
    # Handle case where the token is invalid
    if TOKEN in [None, '', ' ', '  ']:
        print("No valid token was entered")
        print("Using Spacy NER model")
        TOKEN = None

    # Create data object
    data = Data(text, TOKEN, nlp)

    # Emotion analysis
    emotive = []
    expressive = []
    if do_emotion:
        emotive, expressive = extract_expressive_emotive(data, emotion_classifier)

    # Perform NER
    if data.token is None:
        referential = get_ner(data, use_spacy=True)
    else:
        referential = get_ner(data, use_spacy=False)

    # Conative analysis
    conative = get_conative(data)

    return conative, referential, emotive, expressive






# Function to process texts in parallel
def process_texts_in_parallel(texts, do_emotion=True, max_workers=4):
    # Create a list to store the results in the order of the input texts
    results = [None] * len(texts)
    def align_and_resolve_entities(entities,text):
        try:
            entities = align_entities_with_words(text, entities)

            entities = resolve_entities(entities)
        except Exception as e:
                print(f"Error aligning text: {text}\nError: {e} for entities {entities}")
                
        return entities
    # Helper function to process a single text
    def process_single_text(index, text):
        entities_list = extract_entities_from_text(text, do_emotion)
        result = [align_and_resolve_entities(entities,text) for entities in entities_list]
        
        return index, result

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_index = {
            executor.submit(process_single_text, index, text): index
            for index, text in enumerate(texts)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                index, result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing text at index {index}: {e}")
                results[index] = None  

    return results


class CA_Model:
    def __init__(self) -> None:
        pass

    def get_result_count(self,texts,return_counts=True):
        result_count =[]
        results = process_texts_in_parallel(texts,do_emotion=True,max_workers=8)
        for text,result in zip(texts,results):
            result_count.append([*count_conative_ref(result[0],result[1],text,nlp),*count_expressive_emotional(result[2],result[3],text,nlp)])
        if return_counts:
            return results,result_count
        return results



