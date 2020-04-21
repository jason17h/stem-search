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

import typing

import pandas as pd
import numpy as np


app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/css?family=Jacques+Francois+Shadow',
    'https://fonts.googleapis.com/css?family=Amita',
    'https://fonts.googleapis.com/css?family=Josefin+Sans',
    'https://fonts.googleapis.com/css?family=Alegreya+Sans',
    'https://fonts.googleapis.com/css?family=Darker+Grotesque',
    'https://fonts.googleapis.com/css?family=Montserrat',
    'https://fonts.googleapis.com/css?family=Oxygen',
    'https://fonts.googleapis.com/css?family=Playfair+Display+SC',
])

app.layout = html.Div(children=[
    dbc.Row(id='page-header-row', justify='around', children=[
        dbc.Col(id='page-header-col', children=[
            html.H1(id='stemsearch-logo', children=[
                'STEMSearch • ',
                html.Span(id='research-made-easy', children=['Research made easy.']),
            ]),
            # html.Img(src=app.get_asset_url('github-green.png'), style={'height': '2rem'}),
            # html.H4(id='logo-subtitle', children='Created by Jason Huang'),
        ], width=12),
    ]),

    html.Div(id='main-content-div', children=[
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

                    dcc.Tab(label='Recommended', className='data-tab', value='recommended-articles-tab',
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
                                        # TODO: FINISH THIS PARAGRAPH
                                        children="""
                                            If your research is specific to the COVID-19 pandemic, you may find it 
                                            useful to toggle the switch to the CORD-19 data set. CORD-19 stands for 
                                            COVID-19 Open Research Dataset and contains many articles pertaining to 
                                            coronaviruses.
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

                    dcc.Tab(label='My TF–IDF', className='data-tab', value='nlp-dashboard-tab', children=[
                        html.Div(className='data-entry-div', children=[
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
                            html.Br(),
                            html.P("""
                                TF-IDF stands for term frequency-inverse document frequency. It is a quantified measure
                                of how important a term is to a document relative to the rest of the corpus. Use the
                                slider to adjust how many terms to show on the graph to the right.
                            """)
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

                    html.Div(id='nlp-dashboard-div'),
                ])
            ])
        ]),

        dcc.Tabs(children=[
            dcc.Tab(label='Scatter', children=[
                dbc.Row(className='covid-report-row', justify='around', children=[
                    dbc.Col(className='covid-report-plot', width=8, children=[
                        dcc.Graph(
                            id='cases-plot',
                            figure={
                                'data': [{
                                    'x': df_current['population'],#.head(100),
                                    'y': df_current['confirmed'],#.head(100),
                                    'text': df_current['country'],#.head(100),
                                    'mode': 'markers',
                                    'marker': {'color': 'green'},
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
                    dbc.Col(width=4, children=[
                        html.Div(id='covid-report-settings', children=[
                            html.Label('X axis'),
                            dcc.Dropdown(
                                id='select-x-axis-unit',
                                options=[
                                    {'label': 'Population', 'value': 'population'},
                                    {'label': 'Confirmed', 'value': 'confirmed'},
                                ],
                                value='population'
                            ),
                            dcc.RadioItems(
                                id='select-x-axis-scale',
                                options=[
                                    {'label': 'Logarithmic', 'value': 'log'},
                                    {'label': 'Linear', 'value': 'linear'},
                                ],
                                inputStyle={'marginRight': '5px'},
                                labelStyle={'marginRight': '50px'},
                                value='log'
                            ),

                            html.Br(),

                            html.Label('Y axis'),
                            dcc.Dropdown(
                                id='select-y-axis-unit',
                                options=[
                                    {'label': 'Deaths', 'value': 'deaths'},
                                    {'label': 'Confirmed', 'value': 'confirmed'},
                                ],
                                value='confirmed'
                            ),
                            dcc.RadioItems(
                                id='select-y-axis-scale',
                                options=[
                                    {'label': 'Logarithmic', 'value': 'log'},
                                    {'label': 'Linear', 'value': 'linear'},
                                ],
                                inputStyle={'marginRight': '5px'},
                                labelStyle={'marginRight': '50px'},
                                value='log'
                            ),
                        ])
                    ]),
                ]),
            ]),
            dcc.Tab(label='Map', children=[
                dbc.Row(id='select-map-unit-div', justify='around', children=[
                    dbc.Col(width=4, children=[
                        html.Label('Select figure to measure'),
                        dcc.Dropdown(
                            id='select-map-unit',
                            options=[
                                {'label': 'Confirmed', 'value': 'confirmed'},
                                {'label': 'Deaths', 'value': 'deaths'},
                            ],
                            value='confirmed'
                        ),
                    ]),
                ]),
                dbc.Row(className='covid-report-row', justify='around', children=[
                    dbc.Col(id='covid-bubble-map', className='covid-report-plot', width=12, children=[
                        dcc.Graph(
                            figure=go.Figure(go.Scattergeo(
                                lon=df_current['longitude'],
                                lat=df_current['latitude'],
                                text='Country: ' + df_current['country']
                                     + '<br>Confirmed: ' + df_current['confirmed'].astype(str)
                                     + '<br>Deaths: ' + df_current['deaths'].astype(str),
                                visible=True,
                                marker={
                                    'size': list(df_current['confirmed'].values),
                                    'sizeref': 2. * df_current['confirmed'].max() / (20.**2),
                                    'color': 'green',
                                }
                            )),
                            style={'height': '100%', 'width': '100%'}
                        )
                    ]),
                ])
            ]),
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
     State('article-table-body', 'children')]
)
def add_row(n_clicks, title, abstract, table_body):
    """
    Updates the 'My Articles' table with a new article
    :param n_clicks: Number of times the 'Add Article' button has been clicked
    :param title: Title of the new article
    :param abstract: Abstract of the new article
    :param table_body: The body of the table to update
    :return: Updated table body, and empty strings to clear the title & abstract input boxes
    """
    if n_clicks == 1:
        table_body = [html.Tr(children=[html.Td(title), html.Td(abstract)])]
    elif n_clicks > 1:
        table_body.append(html.Tr(children=[html.Td(title), html.Td(abstract)]))

    return table_body, '', ''


@app.callback(
    Output('recommendations-table-body', 'children'),
    [Input('article-table-body', 'children'), Input('database-switch', 'value')],
    [State('add-article-button', 'n_clicks')]
)
def get_recommendations(table_body, db_covid, n_clicks):
    """
    Get the recommended articles based on the updated user articles table
    :param table_body: The recommendations table body
    :param db_covid: Boolean value from the database switch – True if the switch is set to CORD-19
    :param n_clicks: Number of times the add article button has been clicked
    :return:
    """
    if n_clicks == 0:
        return

    # Process the table body into a usable format (e.g. a list)
    articles = []
    for row in table_body:
        tds = row['props']['children']
        articles.append({
            'column-article-title': tds[0]['props']['children'],
            'column-abstract': tds[1]['props']['children']
        })

    # Set the appropriate DataFrame, model and vectorizer
    if db_covid:
        papers, model, vectorizer = covid_papers, covid_model, covid_vectorizer
    else:
        papers, model, vectorizer = arxiv_papers, arxiv_model, arxiv_vectorizer

    # Iterate through the each article and get the recommended articles
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

    # Sort the recommendations based on similarity and filter any duplicated articles
    recommendations = recommendations.sort_values('sim', ascending=False).drop_duplicates(['title', 'abstract']).head(NUM_OF_RECS)

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
                html.Div([
                    dbc.Badge(word, className='top-word-badge', color='success')
                    for word in rec['top_words'].split(',')
                ]),
                html.Div(rec['abstract'])
            ])
        ]) for rec in recommendations.to_dict('records')
    ]


@app.callback(
    [Output('article-table-div', 'style'),
     Output('recommendations-table-div', 'style'),
     Output('nlp-dashboard-div', 'style')],
    [Input('data-tabs', 'value')]
)
def render_data(tab):
    """
    Toggle which data to render on the RHS pane
    :param tab: The tab on the left pane
    :return: Style dictionaries setting the parameters for the desired data and hiding the others
    """
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
    elif tab == 'nlp-dashboard-tab':
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
    """
    Toggle which data set to work with (i.e. arXiv v.s. CORD-19)
    :param db_covid: Boolean value – True means use CORD-19
    :return:
    """
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
    Output('nlp-dashboard-div', 'children'),
    [Input('article-table-body', 'children'), Input('top-n-words-slider', 'value')],
    [State('add-article-button', 'n_clicks')]
)
def render_nlp_dashboard(table_body, n_words, n_clicks):
    """
    Render the visualization of NLP data from the user-inputted articles
    :param table_body: My Articles table body
    :param n_words: The number of words to display on the graph
    :param n_clicks: The number of times the add articles button has been clicked
    :return:
    """
    # Return if the button has not been clicked (to prevent errors upon page startup)
    if n_clicks == 0:
        return

    # Process the table body into a list
    articles = []
    for row in table_body:
        tds = row['props']['children']
        articles.append({
            'column-article-title': tds[0]['props']['children'],
            'column-abstract': tds[1]['props']['children']
        })

    # Get the top n_words words sorted by TF-IDF (greatest to least)
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
            title={'text': 'TF–IDF of my articles'},
            xaxis={'title': 'Word'},
            yaxis={'title': 'TF-IDF'},
        )
    )


@app.callback(
    Output('cases-plot', 'figure'),
    [Input('select-x-axis-unit', 'value'),
     Input('select-x-axis-scale', 'value'),
     Input('select-y-axis-unit', 'value'),
     Input('select-y-axis-scale', 'value')]
)
def update_scatter_plot(x_unit, x_scale, y_unit, y_scale):
    return {
        'data': [{
            'x': df_current[x_unit],  # .head(100),
            'y': df_current[y_unit],  # .head(100),
            'text': df_current['country'],  # .head(100),
            'mode': 'markers',
            'marker': {'color': 'green'},
        }],
        'layout': {
            'title': 'Cases',
            'xaxis': {'title': x_unit.capitalize(), 'type': x_scale},
            'yaxis': {'title': y_unit.capitalize(), 'type': y_scale},
        },
    }


@app.callback(
    Output('covid-bubble-map', 'children'),
    [Input('select-map-unit', 'value')]
)
def update_map(unit):
    return dcc.Graph(
        figure=go.Figure(go.Scattergeo(
            lon=df_current['longitude'],
            lat=df_current['latitude'],
            text='Country: ' + df_current['country']
                 + '<br>Confirmed: ' + df_current['confirmed'].astype(str)
                 + '<br>Deaths: ' + df_current['deaths'].astype(str),
            visible=True,
            marker={
                'size': list(df_current[unit].values),
                'sizeref': 2. * df_current[unit].max() / (20. ** 2),
                'color': 'green',
            }
        )),
        style={'height': '100%', 'width': '100%'}
    )


if __name__ == '__main__':
    app.run_server(debug=True)
