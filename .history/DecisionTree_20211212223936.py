import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from  data import dataPreprocessing,inputData
import base64
from PIL import Image

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


def predictor():
    df_input = inputData()
    model = dt_param_selector()
    y_pred = model.predict(df_input)
    if y_pred == 1:
        goodwatermelon = Image.open("./pics/good.png")
        st.image(goodwatermelon,width=705)
        st.markdown("<center>🍉🍉🍉这瓜甚甜，买一个🍉🍉🍉</center>", unsafe_allow_html=True)
    else:
        file_ = open("./pics/bad2.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" width="705px">',
            unsafe_allow_html=True,
        )
        st.markdown('<center>🔪🔪🔪这瓜不甜，买不得🔪🔪🔪</center>', unsafe_allow_html=True)
    return y_pred,model