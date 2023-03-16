import random
import sys
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn
from sklearn import datasets


def cat_response_cat_predictor(data_set, pred_col, resp_col):
    # Create a crosstab to count occurrences of each combination of categories
    crosstab_data = pd.crosstab(data_set[resp_col], data_set[pred_col])

    # Create a heatmap with the crosstab data
    heatmap = go.Heatmap(
        x=crosstab_data.columns,
        y=crosstab_data.index,
        z=crosstab_data,
        colorscale="Blues",
    )

    # Define the layout of the plot
    layout = go.Layout(
        title="Heatmap", xaxis=dict(title=pred_col), yaxis=dict(title=resp_col)
    )

    # Create the figure
    fig = go.Figure(data=[heatmap], layout=layout)

    # Show the plot
    fig.show()


def cat_resp_cont_predictor(data_set, pred_col, resp_col):
    # Group data together

    # Create histogram using plotly express
    fig_1 = px.histogram(
        data_set, x=pred_col, color=resp_col, nbins=30, histnorm="probability density"
    )
    fig_1.update_layout(
        title="Categorical " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title="Density",
    )
    fig_1.show()

    # Create violin plot using plotly express
    fig_2 = px.violin(
        data_set,
        x=resp_col,
        y=pred_col,
        color=resp_col,
        box=True,
        points="all",
        hover_data=[pred_col],
    )
    fig_2.update_layout(
        title="Categorical " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=resp_col,
        yaxis_title=pred_col,
    )
    fig_2.show()


def cont_resp_cat_predictor(data_set, pred_col, resp_col):
    # Use list comprehension to create hist_data
    hist_data = [
        data_set.loc[data_set[pred_col] == i, resp_col]
        for i in data_set[pred_col].unique()
    ]

    # Use pd.Series.unique instead of pd.DataFrame.unique
    group_labels = data_set[pred_col].unique()

    # Use px.histogram instead of ff.create_distplot
    fig_1 = px.histogram(data_set, x=resp_col, color=pred_col, nbins=30)
    fig_1.update_layout(
        title=f"Continuous {resp_col} vs Categorical {pred_col}",
        xaxis_title=resp_col,
        yaxis_title="Distribution",
    )

    # Combine for loop and zip into a single loop with enumerate
    fig_2_data = [
        go.Violin(
            x=[group_labels[i]] * len(hist_data[i]),
            y=hist_data[i],
            name=group_labels[i],
            box_visible=True,
            meanline_visible=True,
        )
        for i in range(len(hist_data))
    ]
    fig_2 = go.Figure(fig_2_data)
    fig_2.update_layout(
        title=f"Continuous {resp_col} vs Categorical {pred_col}",
        xaxis_title="Groupings",
        yaxis_title=resp_col,
    )

    # Show the plots
    fig_1.show()
    fig_2.show()


def cont_response_cont_predictor(data_set, pred_col, resp_col):
    fig = px.scatter(x=data_set[pred_col], y=data_set[resp_col], trendline="ols")
    fig.update_layout(
        title=f"Continuous {resp_col} vs Continuous {pred_col}",
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.show()


def plot_graphs(df, pred_col, pred_type, resp_col, resp_type):
    if resp_type == "Boolean":
        if pred_type == "Categorical":
            cat_response_cat_predictor(df, pred_col, resp_col)
        elif pred_type == "Continuous":
            cat_resp_cont_predictor(df, pred_col, resp_col)
    elif resp_type == "Continuous":
        if pred_type == "Categorical":
            cont_resp_cat_predictor(df, pred_col, resp_col)
        elif pred_type == "Continuous":
            cont_response_cont_predictor(df, pred_col, resp_col)
    else:
        print("Invalid response type.")


class TestDatasets:
    def __init__(self):
        self.seaborn_data_sets = ["mpg", "tips", "titanic"]
        self.sklearn_data_sets = ["diabetes", "breast_cancer"]
        self.all_data_sets = self.seaborn_data_sets + self.sklearn_data_sets

    TITANIC_PREDICTORS = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "embarked",
        "parch",
        "fare",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
        "class",
    ]

    def get_all_available_datasets(self) -> List[str]:
        return self.all_data_sets

    def get_test_data_set(
        self, data_set_name: str = None
    ) -> Tuple[pd.DataFrame, List[str], str]:

        if data_set_name is None:
            data_set_name = random.choice(self.all_data_sets)
        else:
            if data_set_name not in self.all_data_sets:
                raise Exception(f"Data set choice not valid: {data_set_name}")

        if data_set_name in self.seaborn_data_sets:
            if data_set_name == "mpg":
                data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
                predictors = [
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "origin",
                ]
                response = "mpg"
            elif data_set_name == "tips":
                data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
                predictors = [
                    "total_bill",
                    "sex",
                    "smoker",
                    "day",
                    "time",
                    "size",
                ]
                response = "tip"
            elif data_set_name in ["titanic", "titanic_2"]:
                data_set = seaborn.load_dataset(name="titanic").dropna()
                data_set["alone"] = data_set["alone"].astype(str)
                data_set["class"] = data_set["class"].astype(str)
                data_set["deck"] = data_set["deck"].astype(str)
                data_set["pclass"] = data_set["pclass"].astype(str)
                predictors = self.TITANIC_PREDICTORS
                if data_set_name == "titanic":
                    response = "survived"
                elif data_set_name == "titanic_2":
                    response = "alive"
        elif data_set_name in self.sklearn_data_sets:
            if data_set_name == "boston":
                data = datasets.load_boston()
                data_set = pd.DataFrame(data.data, columns=data.feature_names)
                data_set["CHAS"] = data_set["CHAS"].astype(str)
            elif data_set_name == "diabetes":
                data = datasets.load_diabetes()
                data_set = pd.DataFrame(data.data, columns=data.feature_names)
            elif data_set_name == "breast_cancer":
                data = datasets.load_breast_cancer()
                data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["target"] = data.target
            predictors = data.feature_names
            response = "target"

        # Change category dtype to string
        for predictor in predictors:
            if data_set[predictor].dtype in ["category"]:
                data_set[predictor] = data_set[predictor].astype(str)

        print(f"Data set selected: {data_set_name}")
        data_set.reset_index(drop=True, inplace=True)
        return data_set, predictors, response


def main():
    df_list = []

    # Get all available datasets from TestDatasets class and loop through them
    for test in TestDatasets().get_all_available_datasets():
        # Get dataset, predictors, and response variable for each dataset
        df, predictors, response = TestDatasets().get_test_data_set(data_set_name=test)

        # Append them to the list as a tuple
        df_list.append((df, predictors, response))

    # Get the first dataset, predictors, and response variable
    data_set, predictors, response = df_list[0]

    # Check the response variable type
    if data_set[response].nunique() > 2:
        resp_type = "Continuous"
    else:
        resp_type = "Boolean"

    # Loop through the predictors
    for col in predictors:

        # Check the predictor type
        if isinstance(data_set[col][0], str):
            pred_type = "Categorical"
        else:
            pred_type = "Continuous"

        # Plot graphs
        plot_graphs(
            data_set,
            pred_col=col,
            pred_type=pred_type,
            resp_col=response,
            resp_type=resp_type,
        )

    data_set = df_list[1][0]
    predictors = df_list[1][1]
    response = df_list[1][2]
    if len(data_set[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Boolean"

    for col in predictors:
        if type(data_set[col][0]) == str:
            pred_type = "Categorical"
        else:
            pred_type = "Continuous"
        plot_graphs(
            data_set,
            pred_col=col,
            pred_type=pred_type,
            resp_col=response,
            resp_type=resp_type,
        )

    data_set = df_list[2][0]
    predictors = df_list[2][1]
    response = df_list[2][2]
    if len(data_set[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Boolean"

    for col in predictors:
        if type(data_set[col][0]) == str:
            pred_type = "Categorical"
        else:
            pred_type = "Continuous"
        plot_graphs(
            data_set,
            pred_col=col,
            pred_type=pred_type,
            resp_col=response,
            resp_type=resp_type,
        )

    data_set = df_list[3][0]
    predictors = df_list[3][1]
    response = df_list[3][2]
    if len(data_set[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Boolean"

    for col in predictors:
        if type(data_set[col][0]) == str:
            pred_type = "Categorical"
        else:
            pred_type = "Continuous"
        plot_graphs(
            data_set,
            pred_col=col,
            pred_type=pred_type,
            resp_col=response,
            resp_type=resp_type,
        )

    data_set = df_list[4][0]
    predictors = df_list[4][1]
    response = df_list[4][2]
    if len(data_set[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Boolean"

    for col in predictors:
        if type(data_set[col][0]) == str:
            pred_type = "Categorical"
        else:
            pred_type = "Continuous"
        plot_graphs(
            data_set,
            pred_col=col,
            pred_type=pred_type,
            resp_col=response,
            resp_type=resp_type,
        )


if __name__ == "__main__":
    sys.exit(main())
