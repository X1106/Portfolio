import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
import streamlit as st
warnings.simplefilter('ignore')
font_path = 'TakaoPGothic.ttf'
font_property = FontProperties(fname=font_path)
import plotly.graph_objects as go
import pydeck as pdk












#データの詳細を確認する際に使用いたしました
def check(gulaf):
    print(':::::::::データサイズ:::::\n', gulaf.shape)
    print('::::::::: index:::::::::\n', gulaf.index)
    print('::::::::: カラム名:::::::::\n', gulaf.columns)
    print('::::::::: データ型:::::::::\n', gulaf.dtypes) 


st.title('Portfolio参考資料')
"""
★資料1_pandasを使ってみた
pandas.DataFrameの基本操作
変更と追加/欠損値の変更/データの可視化

★資料2_データ取り込みからデータの可視化
pandas,matplotlibの基本操作
csv読み込み/データの読み込み/データ参照/並び替え/欠損値の処理/
グループ化/データマージデータの可視化/など

※使用した事前処理や可視化に伴うソースコードは
エントリーメールに添付させていただました。
ご確認頂けると幸いです。
"""
st.subheader('資料１.パンダスを使ってみる')
st.text('★簿記二級受験者情報から適当に抜粋した情報の変更/追加')

df={'受験者数':[51727,45173,28572,27854],
     '実受験者数':[39830,35898,2,22626],
     '合格者数':[7255,3091,5440,6932],
     }
df=pd.DataFrame(df,index=['2020.11','2021.02','2021.06','2021.11'])
pd.options.display.float_format = '{:.2f}'.format
df["合格率"]=df["合格者数"]/df['実受験者数']
st.table(df)
st.text('★ 処理後')
pd.options.display.float_format = '{:.2f}'.format
df.loc['2022.02'] = [21974,17448,3057,17.5]
df.iloc[2,1]=22711
df["合格率"]=df["合格者数"]/df['実受験者数']    
df

st.text('★実受験者数と合格者数の割合kを可視化してみる')
fig = plt.figure(figsize=(23,13))
ax = plt.axes()

ax.set_xlabel("(年.月)", fontproperties=font_property,fontsize=20)
ax.set_ylabel("人数", fontproperties=font_property,fontsize=20)

height1 = df['実受験者数']
height2 = df['合格者数']
 
left = np.arange(len(height1))
labels =df.index

width = 0.3
plt.bar(left, height1, color='teal', width=width, align='center',)
plt.bar(left+width, height2, color='navy', width=width, align='center')

plt.xticks(left + width/2, labels,fontsize=18)
plt.show()
st.pyplot(fig)
st.text('')


st.subheader('資料2.データ取り込みからデータの可視化')
"""
2019年度の全国ごみ排出量をCSVから読み込み整えたうえで可視化
"""

garbage = pd.read_csv('./date/ごみ排出量_全国2.csv')
west_garbage = pd.read_csv('./date/ごみ排出量＿西日本.csv')
east_garbage = pd.read_csv('./date/ごみ排出量＿東日本.csv')
lat_lon= pd.read_csv('緯度_経度.csv')

map = pd.merge(garbage,lat_lon, on='県名')
map2=map.set_index('時点')
view = pdk.ViewState(
    longitude=139.691648,
    latitude=35.689185,
    zoom=4,
    pitch=40.5,
)
layer = pdk.Layer(
    "HeatmapLayer",
    data=map2,
    opacity=0.4,
    get_position=["経度", "緯度"],
    threshold=0.3,
    get_weight = 'ごみ総排出量（総量）【ｔ】'
)
layer_map = pdk.Deck(
    layers=layer,
    initial_view_state=view,
)
st.pydeck_chart(layer_map)
show_df = st.checkbox('2019年度.DataFrame')
if show_df == True:
    st.write(map2.loc[['2019年度']])

west= west_garbage.set_index('時点')
east = east_garbage .set_index('時点')
map2019=map2.loc[['2019年度']]
west2019=west.loc[['2019年度']]
sums_west=west2019['ごみ総排出量（総量）【ｔ】'].sum()
east2019=east.loc[['2019年度']]
sums_east=east2019['ごみ総排出量（総量）【ｔ】'].sum()
sums_map2019=map2019['ごみ総排出量（総量）【ｔ】'].sum()
east_west=sums_west+sums_east
height={'ごみ総排出量':[sums_west,sums_east]}
height_df=pd.DataFrame(height,index=['西日本','東日本'])
height_df["割合"]=height_df["ごみ総排出量"]/sums_map2019
east2019["割合"]=east2019['ごみ総排出量（総量）【ｔ】']/sums_east
west2019["割合"]=west2019['ごみ総排出量（総量）【ｔ】']/sums_west

st.subheader('２０１９年ごみ総排出量')
graph_layout=go.Layout(width=900,height=500,margin=dict(l=25, r=50, t=10, b=50, autoexpand=False),yaxis={"range":[1,5000000]},  
)
x=go.Bar(y=map2019['ごみ総排出量（総量）【ｔ】'],x=map2019['県名'])
fig = go.Figure(data=x,layout=graph_layout)
fig.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig)

st.subheader('東日本と西日本のゴミ排出量')
col3, col4= st.columns(2)
with col3:
    fig = plt.figure(figsize=(5,5))
    X = height_df["割合"]
    plt.pie(X,labels=['west','east'],autopct="%1.1f%%")
    plt.show()
    st.pyplot(fig)
with col4:
    """
    """
    """
    """
    """
    """
    """
    """
    """
    ★東西の全国比率 
    """
    """
    """
    st.table(height_df)
west._df = st.checkbox('2019年東日本.DataFrame')
if west._df == True:
    st.write(map2019[['2019年度']])

west._df = st.checkbox('2019年西日本.DataFrame')
if west._df == True:
    st.write(west2019.loc[['2019年度']])
"""
"""
st.text("出典：政府統計の総合窓口(e-Stat),ごみ総排出量（総量）")
st.text("商工会議所2級合格者データ_https://www.kentei.ne.jp/bookkeeping/candidate-data/data_class2")
st.text("本情報は上記情報を加工しています")
