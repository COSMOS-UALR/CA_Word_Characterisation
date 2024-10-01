import spacy
import nltk
import re
from spacy.matcher import Matcher  
from spacy.util import filter_spans
from .utils import  non_performing_verbs,extra_conative







pattern = [{'POS': 'VERB', 'OP': '?'},
           {"POS": "ADV", "OP": "?"},
           {'POS': 'VERB', 'OP': '+'}]

patterns = [
    [{"POS": "VERB"}],  # Match single verbs
    [{"POS": "AUX"}, {"POS": "VERB"}],  # Match auxiliary + main verb
    [{"POS": "VERB"}, {"POS": "PART"}],  # Match verb + particle (e.g., 'look up')
    [{"POS": "AUX"}, {"POS": "ADV", "OP": "?"}, {"POS": "VERB"}],  # Auxiliary + optional adverb + verb
    [{"POS": "VERB"}, {"POS": "ADV", "OP": "?"}],  # Verb + optional adverb (e.g., 'run quickly')
]




import re

def find_word_occurrences(words, sentence,results):
    # Convert the sentence to lowercase for case-insensitive matching
    sentence_lower = sentence.lower()
    
    
    for word in words:
        # Escape the word for regex and create a case-insensitive pattern
        pattern = re.compile(r'\b' + re.escape(word.lower()) + r'\b')
        # Find all matches in the sentence
        matches = pattern.finditer(sentence_lower)
        # Add each match to the results
        for match in matches:
            entry  = {
                'text': sentence[match.start():match.end()],
                'start': match.start(),
                'end': match.end(),
                "entity_group":"conative"
            }
            if entry not in results:
                results.append(entry)
    
    return results



def get_conative(data):

    # instantiate a Matcher instance
    matcher = Matcher(data.nlp.vocab) 
    matcher.add("verb-phrases",  [pattern])

    conative =[]
    idx=0
    def filter_conative(span):
        if len(span)==1:
            if span[0].pos_=='AUX':
                return False
            if span[0].lemma_ in non_performing_verbs:
                return False
        return True 
    for sent in data.doc.sents:
        
        sent_doc = data.nlp(sent.text)
       

        matches = matcher(sent_doc)
        spans = [sent_doc[start:end] for _, start, end in matches if filter_conative(sent_doc[start:end])] 

        spans = filter_spans(spans) 
        for match in spans:
        
            start,end = match.start_char,match.end_char #find_start_end(match.text,sent_doc.text)
            entry = {"text":match.text,"start":idx+start,"end":idx+end,"entity_group":"conative"}
            if entry in conative:
                print(entry,sent_doc,idx)
                raise ValueError("Not a valid result")
                
            conative.append(entry) 
        idx = sent.end_char
    #extra_conative using cognitive words
    find_word_occurrences(extra_conative, data.doc.text,conative)
    return conative
