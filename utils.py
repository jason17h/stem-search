import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import s3fs

import requests
import typing

ACCESS_KEY_ID = os.environ.get('STEMSEARCH_AWS_ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.environ.get('STEMSEARCH_AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.environ.get('STEMSEARCH_AWS_BUCKET_NAME')

fs = s3fs.S3FileSystem(anon=False)

response = requests.get('https://corona-api.com/countries')
country_data = response.json()['data']

paper_columns = ['title', 'authors', 'abstract']
# arxiv_papers = pd.read_parquet('data/arxiv_papers.parquet').loc[:, paper_columns]
# covid_papers = pd.read_parquet('data/cord_papers.parquet').loc[:, paper_columns]
arxiv_papers = pd.read_parquet('s3://stemsearch/arxiv_papers.parquet')
covid_papers = pd.read_parquet('s3://stemsearch/cord_papers.parquet')

NUM_OF_RECS = 20

# with open('model/arxiv_model.pkl', 'rb') as f:
#     arxiv_model = pickle.load(f)
#
# with open('model/tfidf_vectorizer.pkl', 'rb') as f:
#     arxiv_vectorizer = pickle.load(f)
#
# with open('model/covid_model.pkl', 'rb') as f:
#     covid_model = pickle.load(f)
#
# with open('model/covid_tfidf_vectorizer.pkl', 'rb') as f:
#     covid_vectorizer = pickle.load(f)

with fs.open('s3://stemsearch/arxiv_model.pkl') as f:
    arxiv_model = pickle.load(f)

with fs.open('s3://stemsearch/tfidf_vectorizer.pkl') as f:
    arxiv_vectorizer = pickle.load(f)

with fs.open('s3://stemsearch/covid_model.pkl') as f:
    covid_model = pickle.load(f)

with fs.open('s3://stemsearch/covid_tfidf_vectorizer.pkl') as f:
    covid_vectorizer = pickle.load(f)


def get_text(title: typing.Union[pd.Series, str],
             abstract: typing.Union[pd.Series, str]) -> typing.Union[pd.Series, str]:
    return title * 3 + abstract


def process_current_covid_data(data: typing.List[dict]) -> dict:
    """
    Processes the data from COVID-19 API (i.e. extract useful columns, rename columns, etc.)
    :param data: The raw data from the API
    :return: The processed data from the API
    """
    master = dict()

    # Iterate through each dict in the API data, where each dict contains info on one country
    for country in data:
        code = country.pop('code')  # acronym for the country (e.g. Canada is CA, Afghanistan is AF)
        info = country.pop('latest_data')  # contains most recent info (cases, etc.)
        rates = info.pop('calculated')  # we don't need this information

        # Need to change the keys in today's data due to overlap with info
        tmp = country.pop('today')
        today = {'deaths_today': tmp['deaths'], 'confirmed_today': tmp['confirmed']}

        coords = country.pop('coordinates')

        # Update the master dictionary with the information we want
        country.update(info)
        country.update(rates)
        country.update(today)
        country.update(coords)
        
        master[code] = country

    return master


def find_similar_articles(df: pd.DataFrame, model, vectorizer,
                          title: str, abstract: str, cluster_label: int) -> pd.DataFrame:
    """
    Calculate the top NUM_OF_RECS similar articles based on an article inputted by the user using
    cosine similarity.
    :param df: DataFrame containing all the articles (titles, abstracts, authors, etc.) in the data set
    :param model: The model to use (i.e. the arXiv v.s. CORD-19 model)
    :param vectorizer: The vectorizer to use (i.e. the arXiv v.s. CORD-19 vectorizer)
    :param title: Title of the user-inputted article
    :param abstract: Abstract of the user-inputted article
    :param cluster_label: The cluster of the user-inputted article
    :return: DataFrame containing the top NUM_OF_RECS similar articles
    """
    # Get the articles in the same cluster
    this_cluster = model.labels_ == cluster_label
    similar_papers = df.loc[this_cluster]

    # Create the term-document matrices for articles in the same cluster as the inputted article
    this_vector = vectorizer.transform([get_text(title, abstract)])
    that_vector = vectorizer.transform(get_text(similar_papers.loc[:, 'title'], similar_papers.loc[:, 'abstract']))

    # Get the cosine similarity between the matrices and order them from greatest to least
    similarities = cosine_similarity(that_vector, this_vector)
    similar_papers['sim'] = similarities
    similar_papers = similar_papers.sort_values('sim', ascending=False)

    if np.isclose(similar_papers.iloc[0]['sim'], 1):
        # Slice from [1:] since the first value will be the same value that was entered
        return similar_papers.iloc[1:].head(NUM_OF_RECS)
    else:
        return similar_papers.head(NUM_OF_RECS)


# DataFrame containing the COVID-19 API data
df_current = pd.DataFrame(
    process_current_covid_data(country_data)
).T.sort_values('confirmed', ascending=False).rename(columns={'name': 'country'})


def top_n_words(s: typing.Union[pd.Series, typing.List[str]],
                vec, n: int = None) -> typing.List[typing.Tuple[str, float]]:
    """
    Get the top n words in the corpus of articles inputted by the user.
    :param s: List/series of article titles/abstracts inputted by the user
    :param vec: Vectorizer to use (i.e. arXiv vs CORD-19)
    :param n: The number of words to find
    :return: A list of the top n words and their TF-IDFs
    """
    # Get the text processor used by the vectorizer
    analyzer = vec.build_analyzer()

    # Iterate through and process each text (i.e. title/abstract combination) to get the processed words
    # (i.e. no punctuation, etc.)
    processed_words = []
    for text in s:
        processed_words += analyzer(text)
    processed_words = set(processed_words)  # create a set to filter duplicate values

    # Create the bag of words for this text – should be m x n matrix where m is the number of texts passed through
    bow = vec.transform(s)
    words = bow.sum(axis=0)  # taking the sum decreases runtime

    # Get a list of words in the texts and their TF-IDF
    tfidf = [
        (w, words[0, vec.vocabulary_.get(w)])
        for w in processed_words
        if w in vec.vocabulary_ and words[0, vec.vocabulary_.get(w)] != 0
    ]

    tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)  # sort them from greatest to least

    return tfidf[:n]


def top_tfidf(s: str, vec, min_tfidf: float = 0.25) -> str:
    """
    Get the words in a text that exceed a minimum TF-IDF.
    :param s: The text to analyze (i.e. combo of title and abstract)
    :param vec: Vectorizer to use
    :param min_tfidf: The minimum TF-IDF to consider
    :return: A string containing the desired words separated by ','
    """
    analyzer = vec.build_analyzer()
    processed_words = set(analyzer(s))

    bow = vec.transform([s])
    words = bow.sum(axis=0)

    tfidf = [
        (w, words[0, vec.vocabulary_.get(w)])
        for w in processed_words
        if w in vec.vocabulary_ and words[0, vec.vocabulary_.get(w)] > min_tfidf
    ]
    tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
    top_words = [w for w, t in tfidf]

    return ', '.join(top_words)

