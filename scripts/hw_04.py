import os
import random
import sys
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn
import statsmodels
import statsmodels.api
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor


class TestDatasets:
    def __init__(self):
        self.seaborn_data_sets = ["mpg", "tips", "titanic"]
        self.seaborn_data_sets = ["titanic"]
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
    # fig.show()
    fig.write_html(
        file="adarsh21b-plots/" + pred_col + "plot.html", include_plotlyjs="cdn"
    )


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
    # fig_1.show()
    fig_1.write_html(
        file="adarsh21b-plots/" + pred_col + "plot_hist.html", include_plotlyjs="cdn"
    )

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
    # fig_2.show()
    fig_2.write_html(
        file="adarsh21b-plots/" + pred_col + "plot_violin.html", include_plotlyjs="cdn"
    )


def check_directory():
    path = "adarsh21b-plots"
    path_exist = os.path.exists(path)
    if not path_exist:
        os.mkdir(path)


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
    # fig_1.show()
    fig_1.write_html(
        file="adarsh21b-plots/" + pred_col + "plot_hist.html", include_plotlyjs="cdn"
    )
    # fig_2.show()
    fig_2.write_html(
        file="adarsh21b-plots/" + pred_col + "plot_violin.html", include_plotlyjs="cdn"
    )


def cont_response_cont_predictor(data_set, pred_col, resp_col):
    fig = px.scatter(x=data_set[pred_col], y=data_set[resp_col], trendline="ols")
    fig.update_layout(
        title=f"Continuous {resp_col} vs Continuous {pred_col}",
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    # fig.show()
    fig.write_html(
        file="adarsh21b-plots/" + pred_col + "plot.html", include_plotlyjs="cdn"
    )


def logistic_regression_scores(df, response, predictor):

    column = df[predictor]
    y = df[response]
    logistic_regression_model = statsmodels.api.Logit(y, column)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    print(logistic_regression_model_fitted.summary())
    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[0], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[0])

    # Plot the figure
    fig = px.scatter(x=column, y=y, trendline="ols")
    title = f"Variable: {column.name}: (t-value={t_value}) (p-value={p_value})"
    fig.update_layout(
        title=title,
        xaxis_title=f"Variable: {column.name}",
        yaxis_title=f"Response: {y.name}",
    )

    # fig.show()
    fig.write_html(
        file="adarsh21b-plots/" + predictor + "plot.html", include_plotlyjs="cdn"
    )
    return t_value, p_value


def linear_regression_scores(df, response, predictor):
    column = df[predictor]
    y = df[response]
    linear_regression_model = statsmodels.api.OLS(y, column)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[0], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[0])

    # Plot the figure
    fig = px.scatter(x=column, y=y, trendline="ols")
    title = f"Variable: {column.name}: (t-value={t_value}) (p-value={p_value})"
    fig.update_layout(
        title=title,
        xaxis_title=f"Variable: {column.name}",
        yaxis_title=f"Response: {y.name}",
    )

    # fig.show()
    fig.write_html(
        file="adarsh21b-plots/" + predictor + "plot.html", include_plotlyjs="cdn"
    )
    return t_value, p_value


def plot_graphs(df, pred_col, pred_type, resp_col, resp_type):
    if resp_type == "Boolean":
        if pred_type == "Categorical":
            cat_response_cat_predictor(df, pred_col, resp_col)

        elif pred_type == "Continuous":
            cat_resp_cont_predictor(df, pred_col, resp_col)
            t_value, p_value = logistic_regression_scores(df, resp_col, pred_col)
    elif resp_type == "Continuous":
        if pred_type == "Categorical":
            cont_resp_cat_predictor(df, pred_col, resp_col)
        elif pred_type == "Continuous":
            cont_response_cont_predictor(df, pred_col, resp_col)
            t_value, p_value = linear_regression_scores(df, resp_col, pred_col)
    else:
        print("Invalid response type.")


def random_forest(df, continuous_predictors, response):
    X = df[continuous_predictors]
    y = df[response]

    # create a random forest regressor
    model = RandomForestRegressor(random_state=42)

    # Fit the model into the data
    model.fit(X, y)

    importances = model.feature_importances_

    feature_importances = pd.DataFrame(
        {"predictor": continuous_predictors, "importance": importances}
    )

    feature_importances = dict(zip(continuous_predictors, model.feature_importances_))

    print(feature_importances)

    return feature_importances


def plot_dataset(df_list):
    """
    Plot graphs for each dataset in df_list.

    Parameters
    ----------
    df_list : list of tuples
        Each tuple contains a pandas DataFrame, a list of predictor columns,
        and a response column.

    Returns
    -------
    None
    """
    for data_set, predictors, response in df_list:
        # Determine the response type
        if data_set[response].nunique() > 2:
            resp_type = "Continuous"
        else:
            resp_type = "Boolean"

        cont_pred = []
        # Loop through the predictors
        for col in predictors:
            # Determine the predictor type
            if isinstance(data_set[col][0], str) or data_set[col][0] in [True, False]:
                pred_type = "Categorical"
            else:
                pred_type = "Continuous"
                cont_pred.append(col)
            # Plot the graph
            plot_graphs(
                data_set,
                pred_col=col,
                pred_type=pred_type,
                resp_col=response,
                resp_type=resp_type,
            )
        random_forest(data_set, cont_pred, response)


def main():
    check_directory()
    df_list = []

    # Get all available datasets from TestDatasets class and loop through them
    for test in TestDatasets().get_all_available_datasets():
        # Get dataset, predictors, and response variable for each dataset
        df, predictors, response = TestDatasets().get_test_data_set(data_set_name=test)

        # Append them to the list as a tuple
        df_list.append((df, predictors, response))

    # Get the first dataset, predictors, and response variable
    # data_set, predictors, response = df_list[0]
    plot_dataset(df_list)


if __name__ == "__main__":
    sys.exit(main())
