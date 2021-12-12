import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from  data import dataPreprocessing

def dt_param_selector():
    st.sidebar.subheader("请选择模型参数:sunglasses:")
    criterion = st.sidebar.selectbox("criterion", ["gini", "entropy"])
    max_depth = st.sidebar.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.sidebar.number_input(
        "min_samples_split", 1, 20, 2, 1)
    max_features = st.sidebar.selectbox(
        "max_features", [None, "auto", "sqrt", "log2"])

    params = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }

    model = DecisionTreeClassifier(**params)
    df = dataPreprocessing()
    X, y = df[df.columns[:-1]], df["label"]
    model.fit(X, y)
    return model
