from data_clean import dataclean
from similaritymatrix import similarity_matrix
from simtfidf import similarity_matrix_wo_tfidf
from makereco import make_recs
import pandas as pd
import numpy as np
import re
df = pd.read_csv('internshala_dataset.csv')
df=dataclean(df)
# df.insert(0, 'id', range(0, 0 + len(df)))
sim = similarity_matrix(df)
sim.to_csv('internshala_recommendation_matrix.csv', index = True)
sim_1 = similarity_matrix_wo_tfidf(df)
sim_1.to_csv('internshala_recommendation_matrix_wo_tfidf.csv', index = True)
# sim_2 = similarity_matrix_w_lem(df)
# sim_2.to_csv('../data_for_notebooks/internshala_recommendation_matrix_w_lem.csv', index = True)
# sim_3 = similarity_matrix_w_lem_wo_tfidf(df)
# sim_3.to_csv('../data_for_notebooks/internshala_recommendation_matrix_w_lem_wo_tfidf.csv', index = True)
# sim_cat = similarity_matrix_cat(df)
# sim_cat.to_csv('../data_for_notebooks/internshala_recommendation_df_cat.csv', index = True)
sim = pd.read_csv('internshala_recommendation_matrix.csv') 
sim_1 = pd.read_csv('internshala_recommendation_matrix_wo_tfidf.csv')

# setting id column as index
# print(sim.columns)
sim.set_index('id', inplace = True)
sim_1.set_index('id', inplace = True)
# sim_2.set_index('id', inplace = True)
# sim_3.set_index('id', inplace = True)
# sim_cat.set_index('id', inplace = True)
# original internship viewed by the user
df[df.id == 110]
print(make_recs(sim, df, 110, 3))

