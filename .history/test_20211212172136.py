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

# tree = Image.open("./pics/tree.png")
# st.image(tree, "决策树")
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from data import dataPreprocessing
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
# Parameters
n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

# Load data
df = dataPreprocessing()
plt.figure(figsize=(6, 6.5))
for pairidx, pair in enumerate([[1, 0], [1, 3], [1, 4], [1, 5],
 [3, 0], [3, 2], [3, 4], [3, 5]]):
    # We only take the two corresponding features
    X, y = df[df.columns[:-1]].values[:, pair], df["label"]

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    fig=plt.subplot(2, 4, pairidx + 1)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(df.columns[pair[0]])
    plt.ylabel(df.columns[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=df["label"][i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
plt.axis("tight")
# plt.show()
plt.tight_layout()
st.pyplot()


# sio = BytesIO()
# plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
# data = base64.encodebytes(sio.getvalue()).decode()
# src = 'data:image/png;base64,' + str(data)
# st.image(src)
