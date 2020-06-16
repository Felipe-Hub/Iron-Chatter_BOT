import re
import numpy as np
import pandas as pd
import pickle
import json
import keras

from data_cleaning import clean_data
from oov_prep import tagged_n_grams, unknown_words_X, check_and_predict
from pos_tagging import predict_tags

def little_helpers():
    print('potions: re, numpy as np, pandas as pd, pickle, json, nltk, keras, collections')
    print('spells: clean_data, predict_tags, tagged_n_grams, unknown_words_X, check_and_predict')
    pass
    
   
