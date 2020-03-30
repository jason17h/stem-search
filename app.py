import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import requests
import typing

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from styles import *


app = dash.Dash(__name__)

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

app.layout = html.Div(children=[
    html.H1(children='STEMSearch'),
    html.H2(children='Research made easy.'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    html.Div(children=[
        dash_table.DataTable(
            id='article-table',
            columns=[{
                'name': 'Article title',
                'id': 'column-article-title',
            }, {
                'name': 'Abstract',
                'id': 'column-abstract',
            }],
            data=[{'column-article-title': '', 'column-abstract': ''}],
            # editable=True,
            row_deletable=True
        ),

        html.Label('Article title'),
        dcc.Input(id='input-article-title', placeholder='Enter article title', type='text', value=''),

        html.Label('Abstract'),
        dcc.Textarea(id='input-abstract', placeholder='Enter abstract', value=''),

        html.Button('Add article', id='add-article-button', n_clicks=0),
    ], style=container),

    html.Div(children=[
        dash_table.DataTable(
            id='recommendations-table',
            columns=[{
                'name': 'Article title',
                'id': 'column-recommended-article-title',
            }, {
                'name': 'Author(s)',
                'id': 'column-recommended-article-author'
            }, {
                'name': 'Abstract',
                'id': 'column-recommended-abstract',
            }],
            data=[{
                'column-recommended-article-title': '',
                'column-recommended-article-author': '',
                'column-recommended-abstract': ''
            }],
            # data=[{}],
            # editable=True,
            row_deletable=True,
            style_cell={'textAlign': 'left', 'height': '10'},
        ),
    ], style=container),

    html.Div(children=[
        html.Label('Cases by country'),
        dcc.Graph(
            id='cases-graph',
            figure={
                'data': [{
                    'x': df['population'].head(100),
                    'y': df['confirmed'].head(100),
                    'text': df['name'].head(100),
                    'mode': 'markers'
                }],
                'layout': {
                    'xaxis': {'title': 'Population', 'type': 'log'},
                    'yaxis': {'title': 'Confirmed Cases', 'type': 'log'}
                }
            }
        )
    ], style=container)




])


@app.callback(
    [Output('article-table', 'data'),
     Output('input-article-title', 'value'),
     Output('input-abstract', 'value')],
    [Input('add-article-button', 'n_clicks')],
    [State('input-article-title', 'value'),
     State('input-abstract', 'value'),
     State('article-table', 'data')])
def add_row(n_clicks, article, abstract, rows):
    if n_clicks == 1:
        rows = [{'column-article-title': article, 'column-abstract': abstract}]
    elif n_clicks > 1:
        rows.append({'column-article-title': article, 'column-abstract': abstract})
    return rows, '', ''


@app.callback(
    Output('recommendations-table', 'data'),
    [Input('article-table', 'data')],
    [State('add-article-button', 'n_clicks')]
)
def get_arxiv_recommendations(articles, n_clicks):
    print('n_clicks: ', n_clicks)
    if n_clicks == 0:
        return [{
            'column-recommended-article-title': '',
            'column-recommended-article-author': '',
            'column-recommended-abstract': ''
        }]

    print('get arxiv recs')
    recommendations = pd.DataFrame()
    for row in articles:
        title = row['column-article-title']
        abstract = row['column-abstract']
        print('TITLE: {}'.format(title))
        print('ABSTRACT: {}'.format(abstract))
        vec = arxiv_vectorizer.transform([title * 3 + abstract])
        cluster = arxiv_model.predict(vec)

        recommendations = pd.concat(
            [recommendations,
             find_similar_articles(arxiv_papers, arxiv_model, arxiv_vectorizer, title, abstract, cluster[0])],
            ignore_index=True)

        print('+___+++_)I)U(P)OQ@P{W')

    recommendations = recommendations.rename(columns={
        'title': 'column-recommended-article-title',
        'authors': 'column-recommended-article-author',
        'abstract': 'column-recommended-abstract'
    })
    print('here')
    print(recommendations.sort_values('sim', ascending=False).drop(columns='sim').head(num_of_recs).to_dict('records'))
    return recommendations.sort_values('sim', ascending=False).drop(columns='sim').head(num_of_recs).to_dict('records')
    # return [{'column-recommended-article-title': 'asdf', 'column-recommended-article-author': 'asdf', 'column-recommended-abstract': 'adsfg'}]

if __name__ == '__main__':
    app.run_server(debug=True)
