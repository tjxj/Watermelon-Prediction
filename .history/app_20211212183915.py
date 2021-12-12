from PIL import Image
import streamlit as st
from sklearn import preprocessing
from data import getDataSetOrigin, dataPreprocessing, inputData
from visualize import decisionTreeViz,svg_write,plotSurface
import joblib
import base64
from DecisionTree import dt_param_selector,predictor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


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
    st.title("决策树西瓜挑选器")
    st.markdown(collapse_content, unsafe_allow_html=True)

def body():

md_contents():
    predictor()
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
        # viz = decisionTreeViz(model)
        # svg = viz.svg()
        # svg_write(svg)
        ps=plotSurface()