import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import requests
import typing

response = requests.get('https://corona-api.com/countries')
country_data = response.json()['data']

paper_columns = ['title', 'authors', 'abstract']
arxiv_papers = pd.read_parquet('data/papers.parquet').loc[:, paper_columns]
covid_papers = pd.read_parquet('data/cord_papers.parquet').loc[:, paper_columns]

num_of_recs = 20

with open('model/kmeans.pkl', 'rb') as f:
    arxiv_model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    arxiv_vectorizer = pickle.load(f)

with open('model/covid_model.pkl', 'rb') as f:
    covid_model = pickle.load(f)

with open('model/covid_tfidf_vectorizer.pkl', 'rb') as f:
    covid_vectorizer = pickle.load(f)


def process_covid_data(data: typing.List[dict]) -> dict:
    master = dict()
    for country in data:
        code = country.pop('code')
        info = country.pop('latest_data')
        rates = info.pop('calculated')
        # Need to change the keys in today's data due to overlap with info
        tmp = country.pop('today')
        today = {'deaths_today': tmp['deaths'], 'confirmed_today': tmp['confirmed']}
        # We don't care about the coordinates
        country.pop('coordinates')
        country.update(info)
        country.update(rates)
        country.update(today)
        master[code] = country
    return master


def find_similar_articles(df: pd.DataFrame, model, vectorizer,
                          title: str, abstract: str, cluster_label: int) -> pd.DataFrame:
    this_cluster = model.labels_ == cluster_label
    similar_papers = df.loc[this_cluster]
    this_vector = vectorizer.transform([title * 3 + abstract])
    that_vector = vectorizer.transform(similar_papers.loc[:, 'title'] * 3
                                             + similar_papers.loc[:, 'abstract'])

    similarities = cosine_similarity(that_vector, this_vector)
    similar_papers['sim'] = similarities
    similar_papers = similar_papers.sort_values('sim', ascending=False)

    # slice from [1:] since the first value will be the same value that was entered
    if np.isclose(similar_papers.iloc[0]['sim'], 1):
        print('9')
        return similar_papers.iloc[1:].head(num_of_recs)
    else:
        print('9.5')
        return similar_papers.head(num_of_recs)


df = pd.DataFrame(process_covid_data(country_data)).T.sort_values('confirmed', ascending=False)
