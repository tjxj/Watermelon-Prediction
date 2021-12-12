import pandas as pd
import numpy as np
from sklearn import preprocessing
import streamlit as st
import joblib


def getDataSetOrigin():
    dataSet = [
        ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
        ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 1],
        ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
        ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 1],
        ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
        ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 1],
        ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 1],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 1],
        ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0],
        ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0],
        ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0],
        ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0],
        ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0],
        ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0],
    ]
    features = [
        "color",
        "root",
        "knocks",
        "texture",
        "navel",
        "touch",
        "label",
    ]
    dataSet = np.array(dataSet)
    dfOrigin = pd.DataFrame(dataSet, columns=features)
    return dfOrigin


def dataPreprocessing():
    df = getDataSetOrigin()
    for feature in df.columns[0:6]:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
        joblib.dump(le, "./models/" + feature + "_LabelEncoder.model")
    df["label"] = df["label"].astype(int)

    return df


def inputData():
    st.sidebar.subheader("请选择西瓜外观:sunglasses:")
    color = st.sidebar.selectbox("色泽", ("青绿", "乌黑", "浅白"))
    root = st.sidebar.selectbox("根蒂", ("蜷缩", "稍蜷", "硬挺"))
    knocks = st.sidebar.selectbox("敲击", ("浊响", "沉闷", "清脆"))
    texture = st.sidebar.selectbox("纹理", ("清晰", "稍糊", "模糊"))
    navel = st.sidebar.selectbox("脐部", ("凹陷", "稍凹", "平坦"))
    touch = st.sidebar.selectbox("触感", ("硬滑", "软粘"))
    input = [[color, root, knocks, texture, navel, touch]]
    features = ["color", "root", "knocks", "texture", "navel", "touch"]
    np.array(input).reshape(1, 6)
    df_input = pd.DataFrame(input, columns=features, index=None)

    for feature in features[0:6]:
        le = joblib.load("./models/" + feature + "_LabelEncoder.model")
        df_input[feature] = le.transform(df_input[feature])

    return df_input