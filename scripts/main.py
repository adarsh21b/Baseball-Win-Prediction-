import os
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

"""
References for certain libraries and plots:
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
https://plotly.com/python/multiple-axes/
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
https://plotly.com/python/plotly-express/
https://stackoverflow.com/questions/5328556/histogram-matplotlib
"""


def read_data():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
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
    fig = go.Figure()
    # figure 1
    fig = px.scatter(
        df,
        x="sepal length",
        y="sepal width",
        size="petal length",
        template="plotly_dark",
        color="class",
        title="Iris Data Set plot 1 (Sepal Width vs Sepal Length Distribution)",
        size_max=75,
    )
    fig.show()

    fig.write_html(file="adarsh21b-plots/" + "plot1.html", include_plotlyjs="cdn")

    # Figure 2- Violin Plot
    fig2 = go.Figure()
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    for i in species:
        fig2.add_trace(
            go.Violin(
                x=df["class"][df["class"] == i],
                y=df["petal length"][df["class"] == i],
                name=i,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig2.show()
    fig2.write_html(file="adarsh21b-plots/" + "plot2.html", include_plotlyjs="cdn")

    # figure 3- Scatter plot
    fig3 = px.scatter_matrix(
        df,
        dimensions=["sepal length", "sepal width", "petal length", "petal width"],
        color="class",
        template="seaborn",
    )
    fig3.update_layout(
        title="Iris Data set plot2",
        dragmode="select",
        hovermode="closest",
    )
    fig3.show()
    fig3.write_html(file="adarsh21b-plots/" + "plot3.html", include_plotlyjs="cdn")

    # figure 4- Violin PLot
    print(df.head())
    fig4 = px.violin(
        df,
        x="petal width",
        y="petal length",
        color="class",
        box=True,
        title="Iris data set plot 3",
        template="seaborn",
    )
    fig4.show()
    fig4.write_html(file="adarsh21b-plots/" + "plot4.html", include_plotlyjs="cdn")

    # figure 5- Scatter 3d plot
    fig5 = px.scatter_3d(
        df,
        x="sepal length",
        y="sepal width",
        z="petal width",
        color="petal length",
        symbol="class",
    )
    fig3.show()
    fig5.show()
    fig5.write_html(file="adarsh21b-plots/" + "plot5.html", include_plotlyjs="cdn")


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

    # Fit the features to a random forest
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)

    dtree = DecisionTreeClassifier()
    dtree.fit(X, y)
    test_df = df.sample(n=30)
    X_test_orig = test_df[
        ["sepal length", "sepal width", "petal length", "petal width"]
    ].values
    X_test = one_hot_encoder.transform(X_test_orig)
    pipeline = Pipeline(
        [
            ("OneHotEncode", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X, y)
    Pipeline(
        steps=[
            ("OneHotEncode", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    print("Random Forest Score:", pipeline.score(X_test, test_df["class"]))


def mean_graph(df, attribute, flower_type):

    # creating new dataframe with desired class and attribute
    df_new = df[[attribute, "class"]].copy()
    arr = np.asarray(df_new[attribute])
    new_dataframe = df_new.loc[df_new["class"] == flower_type]
    b = np.array(new_dataframe[attribute])

    # Calulating bin value for graph plotting
    countTotal, bin1 = np.histogram(arr, bins=10, range=(min(arr), max(arr)))
    binedge = 0.5 * (bin1[:-1] + bin1[1:])
    countspecific, bin2 = np.histogram(b, bins=bin1, range=(min(b), max(b)))
    ratio = countspecific / countTotal

    mean_variable = sum(countspecific) / sum(countTotal)
    num_array = []
    for i in range(len(binedge)):
        num_array.append(mean_variable)

    # bar graph for attribute
    fig = go.Figure(
        data=go.Bar(
            x=binedge,
            y=countTotal,
            name=attribute,
            # ),
            marker_color="#EB89B5",
            opacity=0.75,
        )
    )
    # Line Graph for average
    fig.add_trace(
        go.Scatter(
            x=binedge,
            y=ratio,
            name="Response",
            yaxis="y2",
            marker=dict(color="crimson"),
        )
    )

    # Mean Line for desired attribute
    fig.add_trace(
        go.Scatter(
            x=binedge,
            y=num_array,
            yaxis="y2",
            mode="lines",
            name="Mean_Response",
        )
    )

    fig.update_layout(
        title_text=flower_type + " analysis on " + attribute,
        bargap=0.2,
        yaxis=dict(
            title=dict(text="Count of each bin"),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            tickmode="auto",
            range=[-0.2, 1.25],
            overlaying="y",
        ),
    ),
    fig.show()
    fig.write_html(
        file="adarsh21b-plots/" + flower_type + "_" + attribute + ".html",
        include_plotlyjs="cdn",
    )


def check_directory():
    path = "adarsh21b-plots"
    path_exist = os.path.exists(path)
    if not path_exist:
        os.mkdir(path)


def main():
    df = read_data()  # Reading data frame
    statistics(df)  # Calling statistics function for different species
    check_directory()  # Checking whether directory is present or not
    graph_plot(df)  # function for different plots on iris dataset
    different_model(df)  # different model for iris dataset

    # Plotting mean of response plot for various characteristics
    # creating dictionary for mean response plot
    d = {
        "Iris-setosa": ["sepal length", "sepal width", "petal length", "petal width"],
        "Iris-virginica": [
            "sepal length",
            "sepal width",
            "petal length",
            "petal width",
        ],
        "Iris-versicolor": [
            "sepal length",
            "sepal width",
            "petal length",
            "petal width",
        ],
    }
    for i in range(4):
        print(d["Iris-setosa"][i])
        mean_graph(df, d["Iris-setosa"][i], "Iris-setosa")
        mean_graph(df, d["Iris-virginica"][i], "Iris-virginica")
        mean_graph(df, d["Iris-versicolor"][i], "Iris-versicolor")
    # code ended for mean response plot


if __name__ == "__main__":
    sys.exit(main())
