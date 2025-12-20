import os

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, MATCH
import plotly.express as px
import pandas as pd
import numpy as np

# External stylesheet
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Data Exploration: Analyze House Price in Hanoi"

# Serve CSS manually (Dash 2.x compatible)
app.css.append_css({"external_url": "/static/stylesheet.css"})

# Read data
df = pd.read_csv("cleaned_data.csv", sep=',')


def display_graphs(n_clicks):
    if n_clicks == 0:
        plot = 'bar'
    else:
        plot = 'scatter'

    return html.Div(
        style={'width': '45%', 'display': 'inline-block',
               'outline': 'thin lightgrey solid', 'padding': 10},
        children=[
            dcc.Graph(
                id={'type': 'dynamic-graph', 'index': n_clicks},
                figure={}
            ),
            dcc.RadioItems(
                id={'type': 'dynamic-choice', 'index': n_clicks},
                options=[
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Scatter Chart', 'value': 'scatter'}
                ],
                value=plot,
                className="card"
            ),
            dcc.Dropdown(
                id={'type': 'dynamic-dpn-ctg', 'index': n_clicks},
                options=[{'label': c.replace('_', ' '), 'value': c}
                         for c in ['House_type', 'Legal_documents',
                                   'No_floor', 'No_bedroom', 'Day_Of_Week']],
                value='House_type',
                clearable=False
            ),
            dcc.Dropdown(
                id={'type': 'dynamic-dpn-num', 'index': n_clicks},
                clearable=False
            )
        ],
        className="wrapper",
    )


# Layout
app.layout = html.Div([
    html.Div(children=[
        html.Div(
            children=[
                html.P("🙀", className="header-emoji"),
                html.H1("House Price in Hanoi Analytics 2020",
                        className="header-title"),
            ],
            className="header",
        )
    ]),
    html.Div(id='container', children=[
        display_graphs(0),
        display_graphs(1)
    ])
])


@app.callback(
    [
        Output({'type': 'dynamic-dpn-num', 'index': MATCH}, 'options'),
        Output({'type': 'dynamic-dpn-num', 'index': MATCH}, 'value')
    ],
    Input({'type': 'dynamic-choice', 'index': MATCH}, 'value')
)
def dropdown_options(radio_value):
    if radio_value == 'bar':
        options = [{'label': x, 'value': x}
                   for x in ['Price', 'Area', 'Length', 'Width']]
        value = 'Price'
    else:
        options = [{'label': x, 'value': x}
                   for x in ['Area', 'Length', 'Width']]
        value = 'Area'

    return options, value


@app.callback(
    Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
    [
        Input({'type': 'dynamic-dpn-ctg', 'index': MATCH}, 'value'),
        Input({'type': 'dynamic-dpn-num', 'index': MATCH}, 'value'),
        Input({'type': 'dynamic-choice', 'index': MATCH}, 'value')
    ]
)
def update_graph(ctg_value, num_value, chart_choice):
    if chart_choice == 'bar':
        dff = df.groupby([ctg_value], as_index=False)[num_value].mean()
        prompt = {
            'Price': 'Average price in million VND per meter square',
            'Area': 'Average area in meter square',
            'Length': 'Average length in meter',
            'Width': 'Average width in meter'
        }
        fig = px.bar(
            dff,
            x=ctg_value,
            y=num_value,
            color_discrete_sequence=px.colors.sequential.Magenta,
            labels={
                ctg_value: ctg_value.replace('_', ' '),
                num_value: prompt[num_value]
            },
            title=f"{ctg_value.replace('_', ' ')} vs the average {num_value.lower()}"
        )
        return fig

    elif chart_choice == 'scatter':
        prompt = {
            'Area': 'Area in meter square',
            'Length': 'Length in meter',
            'Width': 'Width in meter'
        }
        fig = px.scatter(
            df,
            x=num_value,
            y='Price',
            color=ctg_value,
            color_discrete_sequence=px.colors.sequential.Viridis,
            labels={
                ctg_value: ctg_value.replace('_', ' '),
                num_value: prompt[num_value],
                'Price': 'Price (million VND per meter square)'
            },
            title=f"{num_value} vs price by {ctg_value.replace('_', ' ')}"
        )
        return fig


if __name__ == '__main__':
    app.run(debug=True)
