import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc
from dash import html
from flask import Flask
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from utils import convert_to_level

############### Graph Colors ######################

arousal_colors={"0: Low": "skyblue",
                "1: Moderate": "dodgerblue",
                "2: High": "mediumblue",
                "3: Very High": "midnightblue",
                "4: None": '#d4d5d6'}

stress_colors={ "0: Low": "#f7cfcb",
                "1: Moderate": "#e68f85",
                "2: High": "#ba3325",
                "3: Very High": "#7d0f07",
                "4: None": "#d4d5d6"}

active_colors={ "0: Low": "#c5d6ba",
                "1: Moderate": "#8bb872",
                "2: High": "#3b8014",
                "3: Very High": "#112b02",
                "4: None": "#d4d5d6"}

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css'])

global path
path = f'./E4_Data/Family3/' ## here

def update_vizualization(selected_value):
    activity = str(selected_value)
    merged_df, merged_eda_occur, merged_hrv_occur, merged_active_occur = convert_to_level(path, activity)

    fig1 = px.bar(merged_eda_occur, x="Member", y ='count', color='arousal_lv',color_discrete_map=arousal_colors) 
    fig3 = px.bar(merged_hrv_occur, x="Member", y ='count', color='stress_lv',color_discrete_map=stress_colors) 
    fig5 = px.bar(merged_active_occur, x="Member", y ='count', color='active_lv',color_discrete_map=active_colors) 

    figure = [fig1, fig3, fig5]
    for f in figure:
        f.update_layout(barmode='stack',legend={'traceorder':'grouped'}, width=370, height=170)
        f.update_yaxes(visible=False)
        f.update_xaxes(categoryorder='array', categoryarray= ['Father', 'Mother', 'Child'],tickfont={'size': 15})
        f.update_layout(xaxis_title=None)
        f.update_yaxes(visible=False)
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        f.update_layout(
            margin=dict(l=5, r=5, t=0, b=20),
        )
        f.update_layout(legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        for i in range(len(f.data)):
            if f.data[i]['name'] == '4: None':
                f.data[i]['marker']['opacity'] = 0.2
                f['data'][i]['showlegend'] = False


    fig2_1 = px.bar(merged_df, x="datetime", y ='new', color='arousal_lv_x',color_discrete_map=arousal_colors)
    fig2_2 = px.bar(merged_df, x="datetime", y ='new', color='arousal_lv_y',color_discrete_map=arousal_colors)
    fig2_3 = px.bar(merged_df, x="datetime", y ='new', color='arousal_lv_z',color_discrete_map=arousal_colors)
    fig4_1 = px.bar(merged_df, x="datetime", y ='new', color='stress_lv_x',color_discrete_map=stress_colors)
    fig4_2 = px.bar(merged_df, x="datetime", y ='new', color='stress_lv_y',color_discrete_map=stress_colors)
    fig4_3 = px.bar(merged_df, x="datetime", y ='new', color='stress_lv_z',color_discrete_map=stress_colors)
    fig6_1 = px.bar(merged_df, x="datetime", y ='new', color='active_lv_x',color_discrete_map=active_colors)
    fig6_2 = px.bar(merged_df, x="datetime", y ='new', color='active_lv_y',color_discrete_map=active_colors)
    fig6_3 = px.bar(merged_df, x="datetime", y ='new', color='active_lv_z',color_discrete_map=active_colors)

    figures2 = [fig2_1,fig2_2, fig4_1, fig4_2, fig6_1, fig6_2]
    figures3 = [fig2_3, fig4_3,fig6_3]
    for f in figures2:
        f.update_layout(width=370, height=80)
        f.update_layout(showlegend=False) 
        f.update_yaxes(visible=False)
        f.update_layout(xaxis_title=None)
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        f.update_xaxes(tickformat="%H:%M",tickfont={'size': 15,'color': 'white'})
        f.update_layout(
            margin=dict(l=5, r=5, t=0, b=0),
        )
        for i in range(len(f.data)):
            if f.data[i]['name'] == '4: None':
                f.data[i]['marker']['line']['color'] = "gray"
                f.data[i]['marker']['opacity'] = 0.3

    for f in figures3:
        f.update_layout(width=370, height=80)
        f.update_layout(showlegend=False) 
        f.update_yaxes(visible=False)
        f.update_layout(xaxis_title=None)
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        f.update_xaxes(tickformat="%H:%M",tickfont={'size': 15})
        f.update_layout(
            margin=dict(l=5, r=5, t=0, b=20),
        )
        for i in range(len(f.data)):
            if f.data[i]['name'] == '4: None':
                f.data[i]['marker']['line']['color'] = "gray"
                f.data[i]['marker']['opacity'] = 0.3

    return [fig1, fig2_1, fig2_2, fig2_3, fig3, fig4_1, fig4_2, fig4_3, fig5, fig6_1, fig6_2, fig6_3]

    
######### DASH APP LAYOUT ##############
body = dbc.Container([
    dbc.Row(
        html.Div(style={'margin-bottom': '50px'})
    ),
    dbc.Row(
        dbc.Col([
            html.H6('To explore the social and emotional health of our family,', style={'textAlign': 'center', 'font-size': '20px', 'margin-bottom': '0px'}),
            html.H1(children='FamilyScope', style={'textAlign': 'center', 'font-size': '75px','margin-bottom': '20px','margin-top': '0px'})
            
        ])
            
        , style={'margin-top': '20px'}),
    dbc.Row(
        html.Div([
            dbc.Tabs(  
                [
                    dbc.Tab(label="Eating", tab_id="tab-1", tab_style={"marginLeft": "auto",'width': '220px', 'height': '50px'}
                    , label_style={"color": "#6b6b6a",'font-size': '23px','textAlign': 'center','font-family': "Open Sans"}, activeTabClassName="fw-bold"),
                    dbc.Tab(label="Media Watching", tab_id="tab-2"
                    , tab_style={'width': '220px', 'height': '50px'}, label_style={"color": "#6b6b6a", 'font-size': '23px','textAlign': 'center','font-family': "Open Sans"}, activeTabClassName="fw-bold"),
                    dbc.Tab(label="Board Game", tab_id="tab-3"
                    , tab_style={'width': '220px', 'height': '50px'}, label_style={"color": "#6b6b6a", 'font-size': '23px','textAlign': 'center','font-family': "Open Sans"}, activeTabClassName="fw-bold"),
                    dbc.Tab(label="Cleaning", tab_id="tab-4"
                    , tab_style={'width': '220px', 'height': '50px'}, label_style={"color": "#6b6b6a", 'font-size': '23px','textAlign': 'center','font-family': "Open Sans"}, activeTabClassName="fw-bold")
                ],
                id="tabs",
                active_tab="tab-1",
             )
        ], style={'margin-top': '15px'})
    ),
    dbc.Row(children=[
        dbc.Col(
            html.Div(children=[
                html.Video(
                controls = True,
                id = 'video_player',
            )], style={'height':"50%", 'width':"50%",'margin-left': '10px', 'margin-top': '30px'}),
        ),
        dbc.Col([
            dbc.Row([
                html.Div([
                    html.H5('What is FamilyScope?', style={'textAlign': 'left', 'margin-left': '10px', 'margin-top': '10px'}),
                    html.P(['Through physiological(e.g., heart rate, skin conductivity) and behavioral data collected during familial activities,'
                            , 'we provide visualized information about ' ,html.Strong('how aroused, stressed, and active'), ' family members were.']
                    , style={'textAlign': 'left', 'margin-left': '25px', 'margin-top': '10px','margin-right': '20px','margin-bottom': '15px', 'font-size': '15px'})
                ])
            ], style={'margin-left': '5px','margin-right': '5px', 'border-radius': '20px', 'border-style': 'solid', 'border-color': 'lightgray'}),
            dbc.Row([
                html.Div([
                    html.H5('How to use FamilyScope?', style={'textAlign': 'left', 'margin-left': '10px', 'margin-top': '10px'}),
                    html.P(html.Strong('\"Try to explore how our family interacted using video and visualized data!\"')
                    , style={'textAlign': 'center', 'margin-top': '10px', 'font-size': '15px'}),
                    html.P(['Based on the personal data-driven threshold, we show the degree of emotional arousal, stress, and behavioral activeness in ', html.Strong('4 levels(Low/Moderate/High/Very High).'),
                             html.Strong(' The darker the color, the higher the level.')]
                        , style={'textAlign': 'left', 'margin-left': '30px', 'margin-right': '20px','font-size': '14px'}),
                    html.P(['1. Look at individual-level graphs to see how data change over time.', html.Br(),
                            '2. Compare family data with individual- and family-level graphs.', html.Br(),
                            '3. Click a data point on the individual-level graph to see the video at that time.']
                        , style={'textAlign': 'left', 'margin-left': '40px','margin-bottom': '15px','font-size': '14px'})
                    ])
                ]
                , style={'margin-top': '10px','margin-bottom': '10px','margin-left': '5px','margin-right': '5px', 'border-radius': '20px', 'border-style': 'solid', 'border-color': 'lightgray'})
                
            ], style={'width': '99%'})
        ], style={'margin-top':'20px'}),
    dbc.Row([
        dbc.Col([
            dbc.Row(style={'margin-top':'90px'}),
            dbc.Row(
                html.H5(children='Family',style={'textAlign': 'center', 'font-size': '21px', 'margin-top':'70px','margin-bottom':'55px'}), style={ 'margin-left': '3px'}
            ),
            dbc.Row(
                html.H5(children='Fahter',style={'textAlign': 'center', 'margin-top':'20px', 'margin-bottom':'25px'}), style={ 'margin-left': '3px'}
            ),
            dbc.Row(
                html.H5(children='Mother',style={'textAlign': 'center', 'margin-top':'20px', 'margin-bottom':'25px' }), style={ 'margin-left': '3px'}
            ),
            dbc.Row(
                html.H5(children='Child',style={'textAlign': 'center', 'margin-top':'20px' }), style={'margin-bottom': '10px', 'margin-left': '3px'}
            )

        ], width=1),
        dbc.Col([
            dbc.Row(html.H2(children='Emotional Arousal Level', style={'textAlign': 'center', 'margin-top': '10px', 'margin-bottom': '10px','font-size': '30px'}
            ),style={'margin-top': '10px'}),
            html.Div([
                dbc.Row(
                dcc.Graph(
                    id='arousal-overall',
                    config={
                            'displayModeBar': False
                        } ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                            id='arousal-father',
                            config={
                                'displayModeBar': False
                            }
                        ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                            id='arousal-mother',
                            config={
                                'displayModeBar': False
                            }
                        ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                            id='arousal-kid',
                            config={
                                'displayModeBar': False
                            }
                        ), style={'margin-left': 'auto', 'margin-right': 'auto','margin-bottom': '10px'}
                )
            ], style={'background':'white', 'border-radius': '20px', 'box-shadow': 'rgba(0, 0, 0, 0.15) 1.95px 1.95px 2.6px','padding-top': '10px','padding-bottom': '5px'})
           
        ]),
        dbc.Col([
            dbc.Row(
                html.H2(children='Stress Level', style={'textAlign': 'center', 'margin-top': '10px', 'margin-bottom': '10px','font-size': '30px',}
                ),style={'margin-top': '10px'}),
            html.Div([
                dbc.Row(
                dcc.Graph(
                    id='stress-overall',
                    config={
                            'displayModeBar': False
                        }
                    ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                            id='stress-father',
                            config={
                                'displayModeBar': False
                            }
                        ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                            id='stress-mother',
                            config={
                                'displayModeBar': False
                            }
                        ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                            id='stress-kid',
                            config={
                                'displayModeBar': False
                            }
                        ), style={'margin-bottom': '10px','margin-left': 'auto', 'margin-right': 'auto'}
                )
            ], style={'background':'white', 'border-radius': '20px', 'box-shadow': 'rgba(0, 0, 0, 0.15) 1.95px 1.95px 2.6px','padding-top': '10px','padding-bottom': '5px'})
            
            
        ]),
        dbc.Col([
            dbc.Row(
                html.H2(children='Behavioral Activeness Level', style={'textAlign': 'center', 'margin-top': '10px', 'margin-bottom': '10px','font-size': '30px',}
                ),style={'margin-top': '10px'}),
            html.Div([
                dbc.Row(
                dcc.Graph(
                    id='active-overall',
                    config={
                            'displayModeBar': False
                        }
                    ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                        id='active-father',
                        config={
                                'displayModeBar': False
                            }
                        ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                        id='active-mother',
                        config={
                                'displayModeBar': False
                            }
                        ), style={'margin-left': 'auto', 'margin-right': 'auto'}
                ),
                dbc.Row(
                    dcc.Graph(
                        id='active-kid',
                        config={
                                'displayModeBar': False
                            }
                        ), style={'margin-bottom': '10px','margin-left': 'auto', 'margin-right': 'auto'}
                )
            ], style={'background':'white', 'border-radius': '20px', 'box-shadow': 'rgba(0, 0, 0, 0.15) 1.95px 1.95px 2.6px','padding-top': '10px','padding-bottom': '5px'})
            
        ])
    ]),
    dbc.Row(
        html.Div(style={'margin-bottom': '80px'})
    )

    ], fluid='xxl')

app.layout = dbc.Container([body], fluid=True)

@app.callback(
    [Output('arousal-overall','figure'),
    Output('arousal-father','figure'),
    Output('arousal-mother','figure'),
    Output('arousal-kid','figure'),
    Output('stress-overall','figure'),
    Output('stress-father','figure'),
    Output('stress-mother','figure'),
    Output('stress-kid','figure'),
    Output('active-overall','figure'),
    Output('active-father','figure'),
    Output('active-mother','figure'),
    Output('active-kid','figure')],
    [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == 'tab-1':
        return update_vizualization('Eating')
    elif at == 'tab-2':
        return update_vizualization('Media Watching')
    elif at == 'tab-3':
        return update_vizualization('Board Game')
    elif at == 'tab-4':
        return update_vizualization('Cleaning')


@app.callback(
    [Output('video_player', 'src'),
    Output("arousal-father", "clickData"),
    Output("arousal-mother", "clickData"),
    Output("arousal-kid", "clickData"),
    Output("stress-father", "clickData"),
    Output("stress-mother", "clickData"),
    Output("stress-kid", "clickData"),
    Output("active-father", "clickData"),
    Output("active-mother", "clickData"),
    Output("active-kid", "clickData")],
    [Input('tabs', 'active_tab'),
    Input("arousal-father", "clickData"),
    Input("arousal-mother", "clickData"),
    Input("arousal-kid", "clickData"),
    Input("stress-father", "clickData"),
    Input("stress-mother", "clickData"),
    Input("stress-kid", "clickData"),
    Input("active-father", "clickData"),
    Input("active-mother", "clickData"),
    Input("active-kid", "clickData")]
)
def update_video(at, clickData1,clickData2,clickData3,clickData4,clickData5,clickData6, clickData7, clickData8, clickData9):
    if at == 'tab-1':
        activity = 'Eating'
    elif at == 'tab-2':
        activity = 'Media Watching'
    elif at == 'tab-3':
        activity = 'Board Game'
    elif at == 'tab-4':
        activity = 'Cleaning'
    
    list_of_elem = [clickData1,clickData2,clickData3,clickData4,clickData5,clickData6, clickData7, clickData8, clickData9]
    result = True
    for elem in list_of_elem:
        if elem is not None:
            clickData = elem
            result = False

    if result:
        src = f"/static/{activity}.mp4"
        return [src, None, None, None, None, None, None, None, None, None]

    merged_df, _,_,_ = convert_to_level(path, activity)
    init_hour = merged_df['datetime'].min().hour
    init_min = merged_df['datetime'].min().minute
    m = (int(clickData["points"][0]['x'][-5:-3])-init_hour)*60 + int(clickData["points"][0]['x'][-2:])- init_min
    m = m*60
    ts = str(m)
    te = str(m+60)
    src = f"/static/{activity}.mp4#t={ts},{te}"
    return [src, None, None, None, None, None, None, None, None, None]


if __name__ == '__main__':
    app.run_server(debug=True)