import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from  data import dataPreprocessing

def dt_param_selector():
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

def plotSurface():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Parameters
    n_classes = 2
    plot_colors = "ryb"
    plot_step = 0.02

    # Load data
    df = dataPreprocessing()
    plt.figure(figsize=(8,4))
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
