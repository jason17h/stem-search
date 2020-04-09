import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

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

    dbc.Row(justify='between', children=[
        dbc.Col(className='data-col', id='data-entry-col', width=4, children=[
            dcc.Tabs(id='data-tabs', value='my-articles-tab', children=[
                dcc.Tab(label='My articles', value='my-articles-tab', className='data-tab', children=[
                    html.Div(className='data-entry-div', children=[
                        html.Label('Article title'),
                        dbc.Input(id='input-article-title', placeholder='Enter article title', type='text', value=''),
                        html.Br(),

                        html.Label('Abstract'),
                        dbc.Textarea(id='input-abstract', placeholder='Enter abstract', value='', style={'height': '200px'}),
                        html.Br(),

                        dbc.Button('Add article', id='add-article-button', n_clicks=0, className='mr-1', color='light'),
                        html.Br(),
                    ])
                ]),

                dcc.Tab(label='Recommended articles', className='data-tab', value='recommended-articles-tab',
                        children=[
                            html.Div(className='data-entry-div', children=[
                                daq.ToggleSwitch(
                                    id='database-switch',
                                    # color='#FE0000FF',
                                    label=[{'label': 'arXiv', 'style': {'color': '#FE0000FF'}},
                                           {'label': 'CORD-19'}]
                                ),
                                html.Br(),
                                html.P(id='recommended-articles-arxiv-text', style={'color': '#0A5E2AFF'}, children="""
                                    Based on the journals you've referenced so far, here is a list of similar articles
                                    that we think might help you in furthering your research. Feel free to browse the
                                    list, read through the abstracts to see if they fit your research topic, and search
                                    them up if you think they'll be useful! These articles can be found in the arXiv
                                    open-access data archive.
                                """),
                                html.Br(),
                                dbc.Button(
                                    id='arxiv-link',
                                    color='danger',
                                    children='Go to arXiv',
                                    href='https://arxiv.org/',
                                    target='_blank',
                                    style={'backgroundColor': '#FE0000FF', 'borderColor': '#FE0000FF'}
                                ),
                                html.Br(),
                                html.Br(),
                                html.P(
                                    id='recommended-articles-cord-19-text',
                                    children="""
                                    If your research is specific to the COVID-19 pandemic, you may find it useful
                                    """,
                                    style={'borderTop': '#FE0000FF'}
                                ),
                                html.Br(),
                                dbc.Button(
                                    id='cord-19-link',
                                    color='light',
                                    children='Go to CORD-19',
                                    href='https://pages.semanticscholar.org/coronavirus-research',
                                    target='_blank'
                                ),
                            ])
                        ]),

                dcc.Tab(label='Live COVID-19 report', className='data-tab', value='live-covid-report-tab', children=[
                    html.Div(className='data-entry-div', children=[
                        html.Label('Countries'),
                        dcc.Dropdown(
                            id='select-country-dropdown',
                            options=[{'label': 'All countries', 'value': 'ALL'}] + [
                                {'label': row[1]['country'], 'value': row[0]}
                                for row in df_current.loc[:, ['country']].sort_values('country').iterrows()
                            ],
                            value='ALL',
                            multi=True,
                        ),

                        html.Label('X axis'),
                        dcc.Dropdown(
                            options=[
                                {'label': col.capitalize(), 'value': col}
                                for col in ['confirmed', 'country', 'critical', 'deaths', 'population', 'recovered']
                            ],
                            value='population'
                        ),

                        html.Label('Y axis'),
                        dcc.Dropdown(
                            options=[
                                {'label': col.capitalize(), 'value': col}
                                for col in ['confirmed', 'critical', 'deaths', 'population', 'recovered']
                            ],
                            value='confirmed'
                        ),

                        html.Label('Filter'),
                        dcc.Dropdown(
                            options=[
                                {'label': 'Top N', 'value': 'topN'}
                            ],
                            value='confirmed'
                        ),
                    ]),
                ]),
            ]),
        ]),


        dbc.Col(className='data-col', width=7, children=[
            dbc.Card(id='data-display-card', children=[
                html.Div(id='article-table-div', children=[
                    dbc.CardHeader(className='data-display-header', children='My articles'),
                    dbc.CardBody(className='data-display-body', children=[
                        dbc.Table(id='article-table', children=[
                            html.Thead(children=[
                                html.Tr(children=[html.Th('Article title'), html.Th('Abstract')])
                            ]),
                            html.Tbody(id='article-table-body')
                        ]),
                    ]),
                ]),

                html.Div(id='recommendations-table-div', children=[
                    dcc.Loading(color='#FE0000FF', children=[
                        dbc.CardHeader(className='data-display-header', children='Recommended articles'),
                        dbc.CardBody(className='data-display-body', children=[
                            dbc.Table(children=[
                                html.Thead(children=[
                                    html.Tr(children=[html.Th('Article title'), html.Th('Authors'), html.Th('Abstract')])
                                ]),
                                html.Tbody(id='recommendations-table-body')
                            ]),
                        ]),
                    ]),
                ]),

                html.Div(id='live-covid-report-div', children=[
                    dbc.CardHeader(className='data-display-header', children='Cases by country'),
                    dbc.CardBody(className='data-display-body', children=[
                        dcc.Graph(
                            id='cases-graph',
                            figure={
                                'data': [{
                                    'x': df_current['population'],#.head(100),
                                    'y': df_current['confirmed'],#.head(100),
                                    'text': df_current['country'],#.head(100),
                                    'mode': 'markers',
                                    'marker': {'color': '#FE0000FF'},
                                }],
                                'layout': {
                                    'title': 'Cases',
                                    'xaxis': {'title': 'Population', 'type': 'log'},
                                    'yaxis': {'title': 'Confirmed Cases', 'type': 'log'},
                                },
                            },
                            style={'height': '100%', 'width': '100%'},
                        ),
                    ]),
                ]),
            ])
        ])
    ]),


    dbc.Row(id='nlp-dashboard-row', justify='around', children=[
        dbc.Col(width=4, children=[
            html.Label('Top words to display'),
            dcc.Slider(
                id='top-n-words-slider',
                min=1,
                max=100,
                value=25,
                marks={
                    1: {'label': '1'},
                    25: {'label': '25'},
                    50: {'label': '50'},
                    75: {'label': '75'},
                    100: {'label': '100'},
                }
            ),
        ]),
        dbc.Col(id='nlp-dashboard-bar', width=8, children=[
            html.Div(id='nlp-dashboard-bar-graph')
            # dcc.Loading(style={'height': '100%!important'}, children=[
            #     html.Div(id='nlp-dashboard-bar', style={'height': '100%', 'border': '2px red solid'}, children=[])
            # ])
        ]),
    ]),


])


@app.callback(
    [Output('article-table-body', 'children'),
     Output('input-article-title', 'value'),
     Output('input-abstract', 'value')],
    [Input('add-article-button', 'n_clicks')],
    [State('input-article-title', 'value'),
     State('input-abstract', 'value'),
     State('article-table-body', 'children')]
)
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
    Output('recommendations-table-body', 'children'),
    [Input('article-table-body', 'children'), Input('database-switch', 'value')],
    [State('add-article-button', 'n_clicks')]
)
def get_recommendations(table_body, db_covid, n_clicks):
    if n_clicks == 0:
        return

    articles = []
    for row in table_body:
        tds = row['props']['children']
        articles.append({
            'column-article-title': tds[0]['props']['children'],
            'column-abstract': tds[1]['props']['children']
        })

    if db_covid:
        papers, model, vectorizer = covid_papers, covid_model, covid_vectorizer
    else:
        papers, model, vectorizer = arxiv_papers, arxiv_model, arxiv_vectorizer

    recommendations = pd.DataFrame()
    for row in articles:
        title = row['column-article-title']
        abstract = row['column-abstract']
        vec = vectorizer.transform([title * 3 + abstract])
        cluster = model.predict(vec)

        recommendations = pd.concat(
            [recommendations,
             find_similar_articles(papers, model, vectorizer, title, abstract, cluster[0])],
            ignore_index=True
        )

    recommendations = recommendations.sort_values('sim', ascending=False).drop(columns='sim').head(num_of_recs)

    top_texts = get_text(recommendations.loc[:, 'title'], recommendations.loc[:, 'abstract'])
    recommendations['top_words'] = top_texts.apply(lambda s: top_tfidf(s, vectorizer))

    # recommendations = recommendations.rename(columns={
    #     'title': 'column-recommended-article-title',
    #     'authors': 'column-recommended-article-author',
    #     'abstract': 'column-recommended-abstract'
    # })

    return [
        html.Tr(children=[
            html.Td(children=rec['title']),
            html.Td(children=rec['authors']),
            html.Td(children=[
                html.Div([dbc.Badge(word, className='top-word-badge') for word in rec['top_words'].split(',')]),
                html.Div(rec['abstract'])
            ])
        ]) for rec in recommendations.to_dict('records')
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


@app.callback(
    [Output('database-switch', 'label'),
     Output('recommended-articles-arxiv-text', 'style'),
     Output('arxiv-link', 'color'),
     Output('arxiv-link', 'style'),
     Output('recommended-articles-cord-19-text', 'style'),
     Output('cord-19-link', 'color'),
     Output('cord-19-link', 'style')],  # TODO: update button
    [Input('database-switch', 'value')],
)
def toggle_database(db_covid):
    if db_covid:
        return [
            {'label': 'arXiv'},
            {'label': 'CORD-19', 'style': {'color': '#FE0000FF'}}
        ], None, 'light', None, {'color': '#0A5E2AFF'}, 'danger', {'borderColor': '#FE0000FF', 'backgroundColor': '#FE0000FF'}
    else:
        return [
            {'label': 'arXiv', 'style': {'color': '#FE0000FF'}},
            {'label': 'CORD-19'}
        ], {'color': '#0A5E2AFF'}, 'danger', {'borderColor': '#FE0000FF', 'backgroundColor': '#FE0000FF'}, None, 'light', None


@app.callback(
    Output('nlp-dashboard-bar-graph', 'children'),
    [Input('article-table-body', 'children'), Input('top-n-words-slider', 'value')],
    [State('add-article-button', 'n_clicks')]
)
def render_nlp_dashboard(table_body, n_words, n_clicks):
    if n_clicks == 0:
        return

    articles = []
    for row in table_body:
        tds = row['props']['children']
        articles.append({
            'column-article-title': tds[0]['props']['children'],
            'column-abstract': tds[1]['props']['children']
        })

    articles = pd.DataFrame(articles)
    top_words = pd.DataFrame(
        top_n_words(get_text(articles['column-article-title'], articles['column-abstract']), arxiv_vectorizer, n_words)
    ).rename(columns={0: 'word', 1: 'tfidf'}).sort_values(by='tfidf', ascending=False)

    return dcc.Graph(
        style={'height': '100%', 'width': '100%'},
        figure=go.Figure(go.Bar(
            x=top_words['word'],
            y=top_words['tfidf'],
            # orientation='h',
            marker={'color': 'green'},
        )).update_layout(
            # title='NLP Dash',
            xaxis={'title': 'Word'},
            yaxis={'title': 'TF-IDF'},
        )
    )


if __name__ == '__main__':
    app.run_server(debug=True)
