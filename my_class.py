import numpy as np
import pandas as pd
import re
import torch
import sqlalchemy as sa
from sqlalchemy import create_engine, text, types
from sqlalchemy.engine.url import URL
from googletrans import Translator
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util


def preprocess_text(text, stem_lem, lem=bool, stem=bool):
    filtered_text = [re.sub(r"[^\w\s]", '', x.lower()) for x in text.tolist()]
    token_text = [nltk.word_tokenize(x) for x in filtered_text]
    
    if stem:
        stemmer = stem_lem
        prepr_text = [[stemmer.stem(y) for y in x]for x in token_text]
    if lem:
        lemma = stem_lem
        prepr_text = [[lemma.lemmatize(y, pos='v') for y in x]for x in token_text]
    prepr_text = [', '.join(z) for z in prepr_text]
    return prepr_text

def google_translate(text):
        translator = Translator()
        translator.raise_Exception = True
        translated = translator.translate(text, src='ru', dest='en').text
        return translated

class MyGeo:
    
    def __init__(self, test):
        self.engine = create_engine('postgresql+psycopg2://maiiayakusheva:zhuzha271@127.0.0.1/postgres')
        self.query1 = text("""SELECT geonameid, name, country, region FROM geo_df""")
        self.query2 = text("""SELECT * FROM embeddings""")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.test = test

    def get_answer(self):
        with self.engine.begin() as conn:
            query_1 = self.query1
            data = pd.read_sql_query(query_1, conn)
        with self.engine.begin() as conn:
            query_2 = self.query2
            data1 = pd.read_sql_query(query_2, conn)
            target = data.name
            embs = data1.drop('geonameid', axis=1).to_numpy()
            embeddings = torch.Tensor(embs)
            prepr_test = preprocess_text(self.test, WordNetLemmatizer(), lem=True, stem=False)
            test = [google_translate(x) for x in prepr_test]
        table = pd.DataFrame()
        name  = []
        answ = []
        reg = []
        country = []
        cos = []    
        for word in test:
            emb = self.model.encode(word)
            cos_sim = util.cos_sim(embeddings, [emb])
            combinations = []
            for y in range(0, len(cos_sim)):
                combinations.append([y, cos_sim[y]])
            sorted_combinations = sorted(combinations, key=lambda x: x[1], reverse=True)
            name.append(word)
            lst1 = []
            lst2 = []
            lst3 = []
            lst4 = []  
            for j, sc in enumerate(sorted_combinations[0:3]):
                lst1.append(data.name.tolist()[sc[0]])
                lst2.append(sc[1][0])
                lst3.append(data.region.tolist()[sc[0]])
                lst4.append(data.country.tolist()[sc[0]])
            answ.append(lst1)        
            cos.append(lst2)    
            reg.append(lst3)
            country.append(lst4)
            lst1 = []
            lst2 = []
            lst3 = []
            lst4 = []
        table['name'] = name
        table['answer'] = answ
        table['region'] = reg
        table['country'] = country
        table['cos_sim'] = [np.array(x) for x in cos]

        return table
