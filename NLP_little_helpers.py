import re
import numpy as np
import pandas as pd
import pickle
import json

from data_cleaning import clean_data
from oov_prep import tagged_n_grams
from pos_tagging import predict_this

print('Main NLP potions and spells:')
print('clean_data')
print('predict_this')
print('tagged_n_grams')

