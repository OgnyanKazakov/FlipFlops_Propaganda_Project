#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 21:05:58 2019

@author: rmania
"""

import os
import pandas as pd
import numpy as np
path_task1 = os.path.abspath('/Users/rmania/repos/Propaganda Datathon/sample_data/task-1/task1.train.txt')
path_results_task1 = '/Users/rmania/repos/Propaganda Datathon'
df = pd.read_csv(path_task1, sep="\t", names=['article', 'article_id', 'label'])
df.loc[0, 'article']
df.label.value_counts()
df.label = df.label.map({'propaganda': 1, 'non-propaganda': 0})

from textblob import TextBlob
TextBlob(df.loc[200, 'article']).sentiment[0]
df['sentiment_polarity'] = pd.Series([])
df['sentiment_subjectivity'] = pd.Series([])
for article in range(df.shape[0]):
    df.loc[article, 'sentiment_polarity'] = TextBlob(df.loc[article, 'article']).sentiment[0]
    df.loc[article, 'sentiment_subjectivity'] = TextBlob(df.loc[article, 'article']).sentiment[1]
    
df[df.label ==1].head()
df_copy = df.copy()
bins = np.arange(0,1,0.1)
df['subjectivity_binned'] = pd.Series(pd.cut(df['sentiment_subjectivity'], bins).astype(str))
pd.crosstab(df['subjectivity_binned'], df['label']).to_excel(path_results_task1 + '/Subjectivity_Xtab.xlsx')
df['polarity_binned'] = pd.Series(pd.cut(df['sentiment_polarity'], bins).astype(str))
pd.crosstab(df['polarity_binned'], df['label']).to_excel(path_results_task1 + '/Polarity_Xtab.xlsx')


'''
Parts Of Speech Tagging
'''

import en_core_web_sm
from collections import Counter
nlp = en_core_web_sm.load()

df_POS = pd.Series([])
for article in range(df.shape[0])[:2500]:
    df_POS = df_POS.append(pd.DataFrame([Counter([x.label_ for x in nlp(df.loc[article, \
                                                            'article']).ents])]))
#    df_POS.loc[article, 'article_id'] = str(df.loc[article, 'article_id'])

df_POS2 = pd.Series([])
for article in range(df.shape[0])[2501:5000]:
    df_POS2 = df_POS2.append(pd.DataFrame([Counter([x.label_ for x in nlp(df.loc[article, \
                                                            'article']).ents])]))

article = nlp(df.loc[0, 'article'])
article.ents[0].label_
article.ents[0].text
article.ents
labels = [x.label_ for x in article.ents]
pd.DataFrame([Counter(labels)])


'''
Add readability score
'''
import textstat

df['flesch_reading_ease'] = pd.Series([])
df['smog_index'] = pd.Series([])
df['flesch_kincaid_grade'] = pd.Series([])
df['coleman_liau_index'] = pd.Series([])
df['automated_readability_index'] = pd.Series([])
df['dale_chall_readability_score'] = pd.Series([])
df['difficult_words'] = pd.Series([])
df['linsear_write_formula'] = pd.Series([])
df['gunning_fog'] = pd.Series([])
df['text_standard'] = pd.Series([])


for article in range(df.shape[0]):
    df.loc[article, 'flesch_reading_ease'] = textstat.flesch_reading_ease(df.loc[article, 'article'])
    df.loc[article, 'smog_index'] = textstat.smog_index(df.loc[article, 'article'])
    df.loc[article, 'flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(df.loc[article, 'article'])
    df.loc[article, 'coleman_liau_index'] = textstat.coleman_liau_index(df.loc[article, 'article'])
    df.loc[article, 'automated_readability_index'] = textstat.automated_readability_index(df.loc[article, 'article'])
    df.loc[article, 'dale_chall_readability_score'] = textstat.dale_chall_readability_score(df.loc[article, 'article'])
    df.loc[article, 'difficult_words'] = textstat.difficult_words(df.loc[article, 'article'])
    df.loc[article, 'linsear_write_formula'] = textstat.linsear_write_formula(df.loc[article, 'article'])
    df.loc[article, 'gunning_fog'] = textstat.gunning_fog(df.loc[article, 'article'])
    df.loc[article, 'text_standard'] = textstat.text_standard(df.loc[article, 'article'])
    
df.to_csv(path_task1 + '\\DF_with_Readability.csv')

df.head()
for score in ['flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade', 'coleman_liau_index',
       'automated_readability_index', 'dale_chall_readability_score',
       'difficult_words', 'linsear_write_formula', 'gunning_fog',
       'text_standard']:
    pd.crosstab(df[score], df['label']).to_excel(path_results_task1 + '/Readability_' + score + '.xlsx')



