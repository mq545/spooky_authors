#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:18:01 2017

@author: michael
"""

import pandas as pd 
from numpy import mean
import spacy as sp
import os

# Change working directory
os.chdir("/Users/michael/Documents/Kaggle/Spooky_Authors")

# Load SpaCy's english model
nlp = sp.load('en')

# Read in the dialogues
location = "train.csv"
t_df = pd.read_csv(location)

# Process dialogues in SpaCy
t_df['parsed'] = t_df.iloc[:,1].apply(nlp)

# Determine volume of mentions of people
def people_freq(tokenized):
    return len([(ent.text) for ent in tokenized.ents if ent.label_ == "PERSON"])/len(tokenized)

# Apply function to return - PERSON
t_df['ENTITIES'] = t_df.loc[:,['parsed']].iloc[:,0].apply(people_freq)

# Helper function for returning desired spacy POS tags
def get_pos(tokenized):
    return [(token.pos_) for token in tokenized]

# Get POS tags for df
t_df['pos'] = t_df.loc[:,['parsed']].iloc[:,0].apply(get_pos)

# Helper function to measure frequency of PoS for each PoS type
def pos_freq(pos_list,pos_type):
    return pos_list.count(pos_type)/len(pos_list)

# Define PoS that we're interested in
pos_types = ['ADP','NUM','VERB','PART','ADV','PRON','INTJ','PUNCT','CCONJ',
             'SYM','ADJ','NOUN','PROPN','X','DET']

# Apply PoS count function on dataframe
for pos_type in pos_types:
    t_df[pos_type] = t_df['pos'].apply(lambda x: pos_freq(x,pos_type))

# Remove PoS field
del(t_df['pos'])

# Get sentence length
def sent_length(tokenized):
    return len([token.text for token in tokenized])

# Apply function on corpus     
t_df['WCOUNT'] = t_df['parsed'].apply(sent_length)

# Create function to return frequency of specified punct types
def punct_freq(tokenized, punct_type):
    return len([token.text for token in tokenized if 
                token.text == punct_type])/len(tokenized)

# Apply function on corpus to get punct counts
punct_types = {"DQUOTE":'"','SQUOTE': "'","COMMA": ",","PERIOD":".",
               "ELLIPSES": "...", "SEMICOLON":";","COLON":":","QUESTION":"?"}

for key, value in punct_types.items():
    t_df[key] = t_df['parsed'].apply(lambda x: punct_freq(x,value))

# Create function to return frequency of superlatives
def sup_freq(tokenized):
    return len([token.text for token in tokenized if
            token.pos_ == "ADJ" and "est" in token.text.lower()])/len(tokenized)

t_df['SUPERL'] = t_df['parsed'].apply(sup_freq)
    
# Create function for average word length  
def avg_word_length(tokenized):
    return mean([len(token.text) for token in tokenized])

t_df['AVG_WORDLEN'] = t_df['parsed'].apply(avg_word_length)

# Create function for max word length  
def max_word_length(tokenized):
    return max([len(token.text) for token in tokenized])

t_df['MAX_WORDLEN'] = t_df['parsed'].apply(max_word_length)

# Create integer encoding of authors
auth_map = {"HPL":0, "MWS":1,"EAP":2}

def auth_mapper(string):
    return auth_map[string]
t_df['author_num'] = t_df['author'].apply(auth_mapper)

t_df.to_csv("t_df.csv")