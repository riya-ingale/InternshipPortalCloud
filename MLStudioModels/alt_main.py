from data_clean import dataclean
from similaritymatrix import similarity_matrix
from simtfidf import similarity_matrix_wo_tfidf
from alt_makereco import make_recs
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
df = pd.read_csv('internshala_dataset.csv')
df=dataclean(df)
# df.insert(0, 'id', range(0, 0 + len(df)))
sim = similarity_matrix(df)
sim.rename_axis("", axis="columns")
sim.rename_axis("", axis="columns")
sim = sim.rename_axis("", axis="columns")
simrenamed= sim.rename_axis("", axis="index")
simrenamed.insert(0, 'id', range(0, 0 + len(simrenamed)))
newsim=simrenamed.set_index('id')
df[df.id == 110]
print(make_recs(sim, df, 110, 3))