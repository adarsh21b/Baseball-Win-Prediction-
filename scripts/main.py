import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def read_data():
    df = pd.read_csv(
        r"/home/adarsh/Downloads/iris.data",
        names=["sepal length", "sepal width", "petal length", "petal width", "class"],
    )

    df = df.dropna()
    print(df.head(5))
    print(df.info())
    print(df.describe())

    return df


def statistics(df):
    dfr = df.iloc[:, :4]
    print("mean")
    print(np.mean(dfr, axis=0))
    print("median")
    print(np.median(dfr, axis=0))
    print("min")
    print(np.min(dfr, axis=0))
    print("max")
    print(np.max(dfr, axis=0))
    for columndf in dfr:
        print(columndf)
        column = dfr[[columndf]]
        print("quantile 1st\t", "quantile 2nd\t", "quantile 3rd\t", "quantile 4th")
        print(
            np.quantile(column, 0.25),
            "\t\t\t",
            np.quantile(column, 0.50),
            "\t\t\t",
            np.quantile(column, 0.75),
            "\t\t\t",
            np.quantile(column, 1),
        )


def graph_plot(df):
    # Figure 1
    fig = go.Figure()
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    for i in species:
        fig.add_trace(
            go.Violin(
                x=df["class"][df["class"] == i],
                y=df["petal length"][df["class"] == i],
                name=i,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.show()

    # figure 2
    fig2 = px.scatter_matrix(
        df,
        dimensions=["sepal length", "sepal width", "petal length", "petal width"],
        color="class",
        template="seaborn",
    )
    fig2.update_layout(
        title="Iris Data set",
        dragmode="select",
        hovermode="closest",
    )
    fig2.show()

    # figure 3
    print(df.head())
    fig3 = px.violin(
        df,
        x="petal width",
        y="petal length",
        color="class",
        box=True,
        title="Iris data set",
        template="plotly_dark",
    )
    fig3.show()

    # figure 4
    fig = px.scatter(
        df,
        x="sepal length",
        y="sepal width",
        size="petal length",
        template="plotly_dark",
        color="class",
        size_max=60,
    )
    fig.show()


def different_model(df):
    X = df[["sepal length", "sepal width", "petal length", "petal width"]].values
    y = df["class"].values

    # Logistic Regression

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=0).fit(X, y)
    # print(clf.predict(X[:2, :]))
    print("Logistic Regression Score : ", clf.score(X, y))

    # Decision Tree Model
    clf2 = tree.DecisionTreeClassifier()
    clf2 = clf2.fit(X, y)
    # print(clf.predict([[7.7,2.8,6.5,1.8]]))
    print("Decision Tree : ", clf2.score(X, y))

    # Random Forest Model

    X_orig = X
    one_hot_encoder = StandardScaler()
    one_hot_encoder.fit(X_orig)
    X = one_hot_encoder.transform(X_orig)
    print("stop0")

    # Fit the features to a random forest
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)

    dtree = DecisionTreeClassifier()
    dtree.fit(X, y)

    test_df = df.sample(n=30)
    # print_heading("Dummy data to predict")
    # print(test_df)
    X_test_orig = test_df[
        ["sepal length", "sepal width", "petal length", "petal width"]
    ].values
    X_test = one_hot_encoder.transform(X_test_orig)

    # As pipeline
    # print_heading("Model via Pipeline Predictions")
    pipeline = Pipeline(
        [
            ("OneHotEncode", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234))
            # ,("DecisionTree", DecisionTreeClassifier())
        ]
    )
    pipeline.fit(X, y)
    Pipeline(
        steps=[
            ("OneHotEncode", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    print("Random Score:", pipeline.score(X_test, test_df["class"]))


def main():
    df = read_data()  # Reading data frame
    statistics(df)  # Calling statistics function for different species
    graph_plot(df)  # function for 5 different plots on iris dataset
    different_model(df)  # different model for iris dataset


if __name__ == "__main__":
    main()
    sys.exit()
