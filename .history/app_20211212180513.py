from PIL import Image
import streamlit as st
from sklearn import preprocessing
from data import getDataSetOrigin, dataPreprocessing, inputData
from visualize import decisionTreeViz, svg_write
import joblib
import base64
from DecisionTree import dt_param_selector
from test import plotSurface
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# [![Star](https://img.shields.io/github/stars/tjxj/test.svg?logo=github&style=social)](https://github.com/tjxj/test)
# &nbsp[![Follow](https://img.shields.io/twitter/follow/tjxj?style=social)](https://raw.githubusercontent.com/tjxj/100-Days-Of-ML-Code/master/officialaccount.png)
# &nbsp[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee--yellow.svg?logo=buy-me-a-coffee&logoColor=orange&style=social)](https://github.com/tjxj/100-Days-Of-ML-Code/raw/master/buy_me_a_coffee.jpg)
# """
# video_file = open('./pics/华强买瓜.mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes,start_time=0)


# BADGES = """
# <a href="https://gitHub.com/lukasmasuch/streamlit-pydantic" title="Star Repo" target="_blank"><img src="https://img.shields.io/github/stars/lukasmasuch/streamlit-pydantic.svg?logo=github&style=social"></a>
# <a href="https://twitter.com/lukasmasuch" title="Follow on Twitter" target="_blank"><img src="https://img.shields.io/twitter/follow/lukasmasuch.svg?style=social&label=Follow"></a>
# """
# st.markdown(BADGES, unsafe_allow_html=True)

#

st.title("决策树西瓜挑选器")
st.sidebar.subheader("请选择西瓜外观:sunglasses:")

def md_contents():
    collapse_content = """
    <details>
    <summary>Getting started</summary>
    <a href="https://mp.weixin.qq.com/s/OFoTbEcvbe-k3Ys-wTStPA">streamlit 极简入门</a>
    </details>
    
    <details>
    <summary>Additional knowledge</summary>
    <a href="https://mp.weixin.qq.com/s/74ehblIzwe4rmCM6K-ynEA">决策树的可视化</a>
    
    </details>
    
    """
    st.markdown(collapse_content, unsafe_allow_html=True)


md_contents()


# st.image("https://my-wechat.oss-cn-beijing.aliyuncs.com/800_20211207224635.gif",
#     width=705, 
# )

#
df_input = inputData()
st.sidebar.subheader("请选择模型参数:sunglasses:")

model = dt_param_selector()
y_pred = model.predict(df_input)

if y_pred == 1:
    goodwatermelon = Image.open("./pics/good.png")
    st.image(goodwatermelon,width=705)
    st.markdown("<center>🍉🍉🍉这瓜甚甜，买一个🍉🍉🍉</center>", unsafe_allow_html=True)
else:
    file_ = open("./pics/bad.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" >',
        unsafe_allow_html=True,
    )



#     st.image("./pics/bad.gif",
#     width=705, 
# )
    st.markdown('<center>🔪🔪🔪这瓜不甜，买不得🔪🔪🔪</center>', unsafe_allow_html=True)
# 
st.markdown("---")
st.write("Source Code")
with st.expander("data.py", expanded=False):
    with open("data.py", encoding="UTF-8") as f:
        st.code(f.read(), language="python")

with st.expander("DecisionTree.py", expanded=False):
    with open("DecisionTree.py", encoding="UTF-8") as f:
        st.code(f.read(), language="python")

with st.expander("visualize.py", expanded=False):
    with open("visualize.py", encoding="UTF-8") as f:
        st.code(f.read(), language="python")

if st.checkbox("decisionTreeViz"):
    viz = decisionTreeViz(model)
    svg = viz.svg()
    svg_write(svg)