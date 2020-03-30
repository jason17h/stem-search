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


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(id='main-content-div', children=[
    html.H1(children='STEMSearch'),
    html.H2(children='Research made easy.'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dbc.Row(justify='around', children=[
        dbc.Col(className='data-col', id='data-entry-col', width=4, children=[
            dcc.Tabs(id='data-tabs', value='my-articles-tab', children=[
                dcc.Tab(label='My articles', value='my-articles-tab', className='data-tab', children=[
                    html.Div(id='data-entry-div', children=[
                        html.P(" recommended taking a look at some of these articles"),
                        html.Label('Article title'),
                        dbc.Input(id='input-article-title', placeholder='Enter article title', type='text', value=''),
                        html.Br(),

                        html.Label('Abstract'),
                        dbc.Textarea(id='input-abstract', placeholder='Enter abstract', value=''),
                        html.Br(),

                        dbc.Button('Add article', id='add-article-button', n_clicks=0, className='mr-1', color='light'),
                        html.Br(),
                    ])
                ]),
                dcc.Tab(label='Recommended articles', className='data-tab', value='recommended-articles-tab',
                        children=[
                            html.Div(id='recommended-articles-text', children=[
                                html.P("""
                                    Based on the journals you've referred to so far, here is a list of similar articles
                                    that we think might help you in furthering your research. Feel free to browse the
                                    list, read through the abstracts to see if they fit your research topic, and search
                                    them up if you think they'll be useful! These articles can be found in the arXiv
                                    open-access data archive.
                                """),
                                html.Br(),
                                dbc.Button(children='Go to arXiv', href='https://arxiv.org/', target='_blank')
                            ])
                        ]),
                dcc.Tab(label='Live COVID-19 report', className='data-tab', value='live-covid-report-tab', children=[]),
            ]),
        ]),
        dbc.Col(className='data-col', width=7, children=[
            html.Div(id='data-display-div', children=[
                html.Div(id='article-table-div', children=[
                    html.H3('My articles'),
                    dbc.Table(id='article-table', children=[
                        html.Thead(children=[
                            html.Tr(children=[html.Th('Article title'), html.Th('Abstract')])
                        ]),
                        html.Tbody(id='article-table-body')
                    ]),
                ]),
                html.Div(id='recommendations-table-div'),
                html.Div(id='live-covid-report-div', children=[
                    html.H3('Cases by country'),
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
                    ),
                ]),
            ])
        ])
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
    Output('recommendations-table-div', 'children'),
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

    return [
        html.H3('Recommended articles'),
        dbc.Table(children=[
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
    ]


@app.callback(
    [Output('article-table-div', 'style'),
     Output('recommendations-table-div', 'style'),
     Output('live-covid-report-div', 'style')],
    [Input('data-tabs', 'value')]
)
def render_data(tab):
    if tab == 'my-articles-tab':
        return (
            {'maxHeight': '80vh', 'overflowX': 'auto'},
            {'visibility': 'hidden', 'height': '0px', 'overflowX': 'hidden'},
            {'visibility': 'hidden', 'height': '0px', 'overflowX': 'hidden'}
        )
    elif tab == 'recommended-articles-tab':
        return (
            {'visibility': 'hidden', 'height': '0px', 'overflowX': 'hidden'},
            {'maxHeight': '80vh', 'overflowX': 'auto'},
            {'visibility': 'hidden', 'height': '0px', 'overflowX': 'hidden'})
    elif tab == 'live-covid-report-tab':
        return (
            {'visibility': 'hidden', 'height': '0px', 'overflowX': 'hidden'},
            {'visibility': 'hidden', 'height': '0px', 'overflowX': 'hidden'},
            {'maxHeight': '80vh', 'overflowX': 'auto'})


if __name__ == '__main__':
    app.run_server(debug=True)
