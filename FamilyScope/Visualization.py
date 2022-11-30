import os
import pandas as pd
import numpy as np
import math
import csv
from itertools import chain
from datetime import datetime
import matplotlib.pyplot as plt
from biosppy.signals import eda
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc
from dash import html
from flask import Flask
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from utils import convert_to_level

pd.set_option('mode.chained_assignment',  None)

############### Graph Colors ######################

arousal_colors={
                "0: 낮음": "skyblue",
                "1: 적당함": "dodgerblue",
                "2: 높음": "mediumblue",
                "3: 아주 높음": "midnightblue"}

stress_colors={
                "0: 낮음": "#f7cfcb",
                "1: 적당함": "#e68f85",
                "2: 높음": "#ba3325",
                "3: 아주 높음": "#7d0f07"}

active_colors={
                "0: 낮음": "#c5d6ba",
                "1: 적당함": "#8bb872",
                "2: 높음": "#3b8014",
                "3: 아주 높음": "#112b02"}

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css'])

global path
global activities
path = f'./SampleData/'
activities = ['영상 시청', '보드 게임', '청소']

######### DASH APP LAYOUT ##############
body = dbc.Container([
    dbc.Row(
        html.Div(style={'margin-bottom': '50px'})
    ),
    dbc.Row(
        dbc.Col([
            html.H6('우리 가족의 정서적/사회적 건강을 탐색하기 위한,', style={'textAlign': 'center', 'font-size': '20px', 'margin-bottom': '0px'}),
            html.H1(children='FamilyScope', style={'textAlign': 'center', 'font-size': '75px','margin-bottom': '20px','margin-top': '0px'})
            
        ])
            
        , style={'margin-top': '20px'}),
    dbc.Row(
        html.Div([
            dcc.Dropdown(
                id='dropdown-activity',
                value="영상 시청",
                options=['식사', '영상 시청', "보드 게임", '청소'],
                style={'width': '150px',"margin-right": "10px",'float': 'right'}
            ),
            html.H6(['가족 공동 활동:'], style={"margin-top": "5px","margin-right": "20px", 'float': 'right', 'font-size': '18px'})
        ], style={'margin-top': '10px'})
    ),
    dbc.Row(children=[
        dbc.Col(
            html.Div(children=[
                html.Video(
                controls = True,
                id = 'video_player',
            )], style={'margin-left': '10px', 'margin-top': '15px'}),
        ),
        dbc.Col([
            dbc.Row([
                html.Div([
                    html.H5('FamilyScope이란?', style={'textAlign': 'left', 'margin-left': '10px', 'margin-top': '10px'}),
                    html.P(['가족 공동 활동 중 수집된 생체 신호(예. 심박수, 피부 전도도) 와 움직임 신호를 통해'
                            , html.Br(), html.Strong('가족 구성원들이 얼마나 흥분했는지, 스트레스를 받았는지, 활동적이었는지'), '에 대한 정보를',  html.Br(), '시각화하여 제공합니다.']
                    , style={'textAlign': 'left', 'margin-left': '25px', 'margin-top': '10px','margin-right': '20px','margin-bottom': '15px', 'font-size': '15px'})
                ])
            ], style={'margin-left': '5px','margin-right': '5px', 'border-radius': '20px', 'border-style': 'solid', 'border-color': 'lightgray'}),
            dbc.Row([
                html.Div([
                    html.H5('FamilyScope 사용법', style={'textAlign': 'left', 'margin-left': '10px', 'margin-top': '10px'}),
                    html.P(html.Strong('\"동영상과 시각화된 정보를 활용하여 우리 가족이 어떻게 활동했는지 가족이 함께 탐색해보세요!\"')
                    , style={'textAlign': 'center', 'margin-top': '10px', 'font-size': '15px'}),
                    html.P(['개인 데이터 기반의 기준에 따라, 흥분한 정도, 스트레스를 받은 정도, 활동적인 정도를 ', html.Br(), html.Strong('4 단계(낮음/적당함/높음/아주 높음)'),
                        '로 나누어',' 보여줍니다.', html.Strong(' 색이 진할 수록 레벨이 높음'),'을 의미합니다.']
                        , style={'textAlign': 'left', 'margin-left': '30px', 'margin-right': '20px','font-size': '14px'}),
                    html.P(['1. 개인별 그래프를 통해 시간에 따른 변화를 살펴보세요.', html.Br(),
                            '2. 가족의 데이터를 개인별 그래프와 가족 전체 그래프를 통해 비교해보세요.', html.Br(),
                            '3. 개인별 그래프에서 보고 싶은 지점을 선택하면 해당 시간의 동영상을 보실 수 있습니다.']
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
                html.H5(children='가족 전체',style={'textAlign': 'center', 'font-size': '21px', 'margin-top':'65px','margin-bottom':'55px'}), style={ 'margin-left': '3px'}
            ),
            dbc.Row(
                html.H5(children='아빠',style={'textAlign': 'center', 'margin-top':'35px', 'margin-bottom':'35px'}), style={ 'margin-left': '3px'}
            ),
            dbc.Row(
                html.H5(children='엄마',style={'textAlign': 'center', 'margin-top':'35px', 'margin-bottom':'35px' }), style={ 'margin-left': '3px'}
            ),
            dbc.Row(
                html.H5(children='아이',style={'textAlign': 'center', 'margin-top':'35px', 'margin-bottom':'35px' }), style={'margin-bottom': '10px', 'margin-left': '3px'}
            )

        ], width=1),
        dbc.Col([
            dbc.Row(html.H2(children='감정 흥분 지수', style={'textAlign': 'center', 'margin-top': '10px', 'margin-bottom': '10px','font-size': '33px'}
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
                html.H2(children='스트레스 지수', style={'textAlign': 'center', 'margin-top': '10px', 'margin-bottom': '10px','font-size': '33px',}
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
                html.H2(children='활동성 지수', style={'textAlign': 'center', 'margin-top': '10px', 'margin-bottom': '10px','font-size': '33px',}
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
    [Input('dropdown-activity', 'value'),
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
def update_video(selected_value, clickData1,clickData2,clickData3,clickData4,clickData5,clickData6, clickData7, clickData8, clickData9):
    activity = selected_value
    list_of_elem = [clickData1,clickData2,clickData3,clickData4,clickData5,clickData6, clickData7, clickData8, clickData9]
    result = True
    for elem in list_of_elem:
        if elem is not None:
            clickData = elem
            result = False

    if result:
        src = f"/static/{activity}.mp4"
        return [src, None, None, None, None, None, None, None, None, None]
        
    merged_df, _,_,_ = convert_to_level(path, activities, activity)
    init_hour = merged_df['datetime'].min().hour
    init_min = merged_df['datetime'].min().minute
    m = (int(clickData["points"][0]['x'][-5:-3])-init_hour)*60 + int(clickData["points"][0]['x'][-2:])- init_min
    m = m*60
    ts = str(m)
    te = str(m+60)
    src = f"/static/{activity}.mp4#t={ts},{te}"
    return [src, None, None, None, None, None, None, None, None, None]

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
    [Input('dropdown-activity', 'value')],)
def update_vizualization(selected_value):
    activity = str(selected_value)
    merged_df, merged_eda_occur, merged_hrv_occur, merged_active_occur = convert_to_level(path, activities, activity)

    fig1 = px.bar(merged_eda_occur, x="Member", y ='count', color='arousal_lv',color_discrete_map=arousal_colors) 
    fig3 = px.bar(merged_hrv_occur, x="Member", y ='count', color='stress_lv',color_discrete_map=stress_colors) 
    fig5 = px.bar(merged_active_occur, x="Member", y ='count', color='active_lv',color_discrete_map=active_colors) 

    figure = [fig1, fig3, fig5]
    for f in figure:
        f.update_layout(barmode='stack', width=370, height=180)
        f.update_xaxes(categoryorder='array', categoryarray= ['아빠', '엄마', '아이'],tickfont={'size': 13})
        f.update_layout(xaxis_title=None)
        f.update_yaxes(visible=False)
        f.update_layout(
            margin=dict(l=5, r=5, t=0, b=20),
        )
        f.update_layout(legend=dict(
            title="",
            orientation="h",
            yanchor="top",
            y=1.22,
            xanchor="right",
            x=1
        ))


    
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
        f.update_layout(width=370, height=100)
        f.update_layout(showlegend=False) 
        f.update_yaxes(visible=False)
        f.update_layout(xaxis_title=None)
        f.update_xaxes(tickformat="%H:%M",tickfont={'color': 'white'})
        f.update_layout(
            margin=dict(l=5, r=5, t=0, b=0),
        )
    for f in figures3:
        f.update_layout(width=370, height=100)
        f.update_layout(showlegend=False) 
        f.update_yaxes(visible=False)
        f.update_layout(xaxis_title=None)
        f.update_xaxes(tickformat="%H:%M",tickfont={'size': 13})
        f.update_layout(
            margin=dict(l=5, r=5, t=0, b=20),
        )

    return [fig1, fig2_1, fig2_2, fig2_3, fig3, fig4_1, fig4_2, fig4_3, fig5, fig6_1, fig6_2, fig6_3]
if __name__ == '__main__':
    app.run_server(debug=True)