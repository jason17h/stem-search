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


{'props':
     {'children': [
         {'props': {'children': "'Infectious diarrhea: Pathogenesis and risk factors'"}, 'type': 'Td', 'namespace': 'dash_html_components'},
         {'props': {'children': 'Abstract Middle-aged female identical twins, one of whom had systemic lupus erythematosus (SLE), were evaluated for \nimmunologic reactivity to previous antigenic challenges, including primary immunization with a foreign antigen, \nkeyhole limpet hemocyanin (KLH). These two women had lived together for all of their 58 years and neither was \nreceiving anti-inflammatory or immunosuppressive drugs at the time of these studies. Both twins demonstrated \ncomparable 7S and 198 humoral antibody response to KLH, as well as similar viral antibody titers. However, the twin\nwith SLE was anergic to common antigens, streptokinase-streptodornase, Trichophyton and Candida; furthermore delayed \nhypersensitivity to KLH did not develop after immunization. This observed discrepancy between humoral and cellular \nimmunity in genetically similar subjects may be significant in the pathogenesis of SLE.'}, 'type': 'Td', 'namespace': 'dash_html_components'}
     ]}, 'type': 'Tr', 'namespace': 'dash_html_components'
 }

