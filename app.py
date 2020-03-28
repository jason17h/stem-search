import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import requests
import typing
import pandas as pd

app = dash.Dash(__name__)

response = requests.get('https://corona-api.com/countries')
country_data = response.json()['data']


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


df = pd.DataFrame(process_covid_data(country_data)).T.sort_values('confirmed', ascending=False)

app.layout = html.Div(children=[
    html.H1(children='STEMSearch'),
    html.H2(children='Research made easy.'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    # html.Label('Citation format'),
    # dcc.Dropdown(
    #     id='citation-format',
    #     options=[
    #         {'label': 'MLA', 'value': 'MLA'},
    #         {'label': 'APA', 'value': 'APA'},
    #         {'label': 'Chicago', 'value': 'CHI'}
    #     ],
    #     value='MLA',
    # ),

    html.Div(children=[
        dash_table.DataTable(
            id='adding-rows-table',
            columns=[{
                'name': 'Article title',
                'id': 'column-article-title',
            }, {
                'name': 'Abstract',
                'id': 'column-abstract',
            }],
            data=[{'column-article-title': 'Sample title', 'column-abstract': 'Sample Abstract'}],
            editable=True,
            row_deletable=True
        ),

        html.Button('Add Row', id='editing-rows-button', n_clicks=0),
    ]),

    html.Button('Find similar articles', id='find-related-button', n_clicks=0),

    # html.Label('Article title: '),
    # dcc.Input(id='article-title-input', type='text', placeholder='Enter article title'),
    # html.Label('Abstract: '),
    # dcc.Input(id='abstract-input', type='text', placeholder='Enter abstract (optional)'),
    # html.Button('Add article', id='submit-article-button'),

    html.Div(children=[
        html.Label('Cases'),
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
    Output('adding-rows-table', 'data'),
    [Input('editing-rows-button', 'n_clicks')],
    [State('adding-rows-table', 'data'),
     State('adding-rows-table', 'columns')])
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows

if __name__ == '__main__':
    app.run_server(debug=True)
