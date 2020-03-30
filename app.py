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

app.layout = dbc.Container(children=[
    html.H1(children='STEMSearch'),
    html.H2(children='Research made easy.'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Tabs(id='table-tabs', value='my-articles-tab', children=[
        dcc.Tab(label='My articles', value='my-articles-tab', children=[
            html.Label('Article title'),
            dbc.Input(id='input-article-title', placeholder='Enter article title', type='text', value=''),

            html.Label('Abstract'),
            dbc.Textarea(id='input-abstract', placeholder='Enter abstract', value=''),

            dbc.Button('Add article', id='add-article-button', n_clicks=0, className='mr-1', color='light'),

            dbc.Table(id='article-table', children=[
                html.Thead(children=[
                    html.Tr(children=[html.Th('Article title'), html.Th('Abstract')])
                ]),
                html.Tbody(id='article-table-body')
            ])



            # dash_table.DataTable(
            #     id='article-table',
            #     columns=[{
            #         'name': 'Article title',
            #         'id': 'column-article-title',
            #     }, {
            #         'name': 'Abstract',
            #         'id': 'column-abstract',
            #     }],
            #     data=[{'column-article-title': '', 'column-abstract': ''}],
            #     row_deletable=True
            # ),
        ]),
        dcc.Tab(label='Recommended articles', value='recommended-articles-tab',
                children=[html.Div(id='recommendations-table')]),
    ]),

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
    [Output('article-table-body', 'children'),
     Output('input-article-title', 'value'),
     Output('input-abstract', 'value')],
    [Input('add-article-button', 'n_clicks')],
    [State('input-article-title', 'value'),
     State('input-abstract', 'value'),
     State('article-table-body', 'children')])
def add_row(n_clicks, article, abstract, table_body):
    # if n_clicks == 1:
    #     rows = [{'column-article-title': article, 'column-abstract': abstract}]
    # elif n_clicks > 1:
    #     rows.append({'column-article-title': article, 'column-abstract': abstract})
    # return rows, '', ''
    if n_clicks == 1:
        table_body = [html.Tr(children=[html.Td(article), html.Td(abstract)])]
    elif n_clicks > 1:
        table_body.append(html.Tr(children=[html.Td(article), html.Td(abstract)]))

    return table_body, '', ''


@app.callback(
    # Output('recommendations-table', 'data'),
    Output('recommendations-table', 'children'),
    [Input('article-table-body', 'children')],
    [State('add-article-button', 'n_clicks')]
)
def get_arxiv_recommendations(table_body, n_clicks):
    if n_clicks == 0:
        return

    articles = []
    for row in table_body:
        print('ROW:')
        print(row)
        tds = row['props']['children']
        articles.append({
            'column-article-title': tds[0]['props']['children'],
            'column-abstract': tds[1]['props']['children']
        })

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
