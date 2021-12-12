# 本文用到的库
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import base64
import streamlit as st
from sklearn import preprocessing
from dtreeviz.trees import *
from data import getDataSetOrigin,dataPreprocessing
import joblib
from DecisionTree import dt_param_selector


def decisionTreeViz(clf):
    df = dataPreprocessing()
    X, y = df[df.columns[:-1]], df["label"]
    viz = dtreeviz(
        clf,
        X,
        y,
        orientation="LR",
        target_name="label",
        feature_names=df.columns[:-1],
        class_names=["good", "bad"],  # need class_names for classifier
    )

    return viz


def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = (
        f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    )
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True, width=800, caption="决策树")
