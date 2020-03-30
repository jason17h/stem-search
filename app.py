import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from utils import *

import requests
import typing

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from styles import *


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    html.H1(children='STEMSearch'),
    html.H2(children='Research made easy.'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dbc.Container([
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
            row_deletable=True
        ),

        html.Label('Article title'),
        dcc.Input(id='input-article-title', placeholder='Enter article title', type='text', value=''),

        html.Label('Abstract'),
        dcc.Textarea(id='input-abstract', placeholder='Enter abstract', value=''),

        html.Button('Add article', id='add-article-button', n_clicks=0),
    ]),

    # html.Div(children=[
    #     dash_table.DataTable(
    #         id='recommendations-table',
    #         columns=[{
    #             'name': 'Article title',
    #             'id': 'column-recommended-article-title',
    #         }, {
    #             'name': 'Author(s)',
    #             'id': 'column-recommended-article-author'
    #         }, {
    #             'name': 'Abstract',
    #             'id': 'column-recommended-abstract',
    #         }],
    #         data=[{
    #             'column-recommended-article-title': '',
    #             'column-recommended-article-author': '',
    #             'column-recommended-abstract': ''
    #         }],
    #         # data=[{}],
    #         # editable=True,
    #         row_deletable=True,
    #         style_cell={'textAlign': 'left', 'height': '10'},
    #     ),
    # ], style=container),

    dbc.Container(id='recommendations-table'),

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
    ])




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
    # Output('recommendations-table', 'data'),
    Output('recommendations-table', 'children'),
    [Input('article-table', 'data')],
    [State('add-article-button', 'n_clicks')]
)
def get_arxiv_recommendations(articles, n_clicks):
    if n_clicks == 0:
        # return [{
        #     'column-recommended-article-title': '',
        #     'column-recommended-article-author': '',
        #     'column-recommended-abstract': ''
        # }]
        return

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

    recommendations = recommendations.rename(columns={
        'title': 'column-recommended-article-title',
        'authors': 'column-recommended-article-author',
        'abstract': 'column-recommended-abstract'
    }).sort_values('sim', ascending=False).drop(columns='sim').head(num_of_recs).to_dict('records')

    # return recommendations
    return dbc.Table(children=[
        html.Thead(children=[
            html.Tr(children=[html.Th('Article title'), html.Th('Authors'), html.Th('Abstract')])
        ]),
        html.Tbody(
            children=[
                html.Tr(children=[
                    html.Td(children=[value]) for key, value in r.items()
                ]) for r in recommendations
            ]
        )
    ])


if __name__ == '__main__':
    app.run_server(debug=True)
