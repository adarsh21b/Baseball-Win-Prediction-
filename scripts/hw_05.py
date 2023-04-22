import os
import sys
import warnings

import numpy as np
import pandas
import pandas as pd
import plotly.io as pio
import sqlalchemy
import statsmodels
import statsmodels.api
from plotly import express as px
from plotly import graph_objects as go
from scipy import stats
from scipy.stats import pearsonr
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
References:
https://en.wikipedia.org/wiki/Baseball_statistics
https://teaching.mrsharky.com/sdsu_fall_2020_lecture11.html#/9/1
https://www.mlb.com/glossary/standard-stats/walk
https://www.mlb.com/glossary/standard-stats/walks-and-hits-per-inning-pitched
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
https://plotly.com/python/multiple-axes/
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
https://plotly.com/python/plotly-express/
https://stackoverflow.com/questions/5328556/histogram-matplotlib
https://pandas.pydata.org/docs/reference/api/pandas.cut.html
https://www.geeksforgeeks.org/pandas-cut-method-in-python/#
https://plotly.com/python/heatmaps/
https://www.w3schools.com/sql/sql_case.asp

"""


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T
    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow
    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T
    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pandas.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def check_response(df, response):
    if len(df[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(df[response].value_counts()) == 2:
        resp_type = "Categorical"

    return resp_type


def correlation_matrix(x, y, score, analysis_name):
    figure = go.Figure(
        data=go.Heatmap(
            x=x.values,
            y=y.values,
            z=score,
            zmin=0.0,
            zmax=1.0,
            type="heatmap",
            colorscale="RdBu",
            text=round(score, 6),
            texttemplate="%{text}",
        ),
        layout={
            "title": f"{analysis_name}",
        },
    )

    return pio.to_html(figure, include_plotlyjs="cdn")


def check_predictor(df, predictor):
    print("inside check predictor")
    print(predictor)
    print(df[predictor].dtype)

    if df[predictor].dtype in ["category", "object", "bool"]:
        pred_type = "Categorical"
    else:
        pred_type = "Continuous"

    return pred_type


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def random_forest_value(df, response, predictors):
    X = df[predictors]
    y = response

    # create a random forest regressor
    model = RandomForestRegressor(random_state=42, n_estimators=150)

    # Fit the model into the data
    model.fit(X, y)

    importances = model.feature_importances_

    feature_importances = pd.DataFrame(
        {"predictor": predictors, "importance": importances}
    )

    feature_importances = dict(zip(predictors, model.feature_importances_))

    print("inside random forest")
    # print(feature_importances)

    return feature_importances


def check_directory():
    path = "adarsh21b-plots"
    path_exist = os.path.exists(path)
    if not path_exist:
        os.mkdir(path)


def logistic_reg(df, response, predictor, predtype, resp, file=None):
    print("Inside Logistic Regression")

    if predictor.dtype not in ["category", "object", "bool"]:
        predict = statsmodels.api.add_constant(predictor)
        logistic_regression_model = statsmodels.api.Logit(response, predict)
        logistic_regression_model_fitted = logistic_regression_model.fit()

        # Get the stats
        t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

        group_labels = df[resp].unique().astype(str)
        hist_data = []

        for label in group_labels:
            hist_data.append(np.array(df[df[resp].astype(str) == label][predtype]))

        fig_2 = go.Figure()

        for curr_hist, curr_group in zip(hist_data, group_labels):
            print(f"curr_hist = {curr_hist}")
            print(f"curr_group = {curr_group}")
            fig_2.add_trace(
                go.Violin(
                    x=np.repeat(curr_group, len(curr_hist)),
                    y=curr_hist,
                    name=curr_group,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig_2.update_layout(
            title=f"Continuous Response {resp} by Categorical Predictor {predtype}",
            xaxis_title="Groupings",
            yaxis_title="Response",
        )

        # fig_2.show()
        file = f"adarsh21b-plots/{resp}vs{predtype}plot_violin.html"
        fig_2.write_html(file=file, include_plotlyjs="cdn")

        return t_value, p_value, file


def linear_reg(df, response, predictor, predtype, resp, file=None):
    print("Inside linear Regression")
    pred = statsmodels.api.add_constant(predictor)
    linear_regression_model = statsmodels.api.OLS(response, pred)
    linear_regression_model_fitted = linear_regression_model.fit()
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    print(predictor.dtype)
    if predictor.dtype in ["category", "object", "bool"]:

        group_labels = predictor.unique()
        hist_data = []

        for label in group_labels:
            hist_data.append(np.array(df[df[predictor] == label][response]))

        fig_2 = go.Figure()
        for curr_hist, curr_group in zip(hist_data, group_labels):
            fig_2.add_trace(
                go.Violin(
                    x=np.repeat(curr_group, len(curr_group)),
                    y=curr_hist,
                    name=curr_group,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig_2.update_layout(
            title=f"Continuous Response {resp} by Categorical Predictor {predtype}",
            xaxis_title="Groupings",
            yaxis_title="Response",
        )

        # fig_2.show()

        file = f"adarsh21b-plots/{resp}vs{predtype}plot_violin.html"
        file = fig_2.write_html(file=file, include_plotlyjs="cdn")

    else:
        fig = px.scatter(x=predictor, y=response, trendline="ols")
        title = "Continous Response by Continous Predictor"
        fig.update_layout(
            title=title,
            xaxis_title=f"{predictor.name}",
            yaxis_title=f"{response.name}",
        )
        # fig.show()

        file = f"adarsh21b-plots/{resp}vs{predtype}scatter.html"
        fig.write_html(file=file, include_plotlyjs="cdn")

    return t_value, p_value, file


def continuous_cont(df, continous_predictors, response_df, response):
    combinations = []
    for i in range(len(continous_predictors)):
        for j in range(len(continous_predictors)):
            combinations.append((continous_predictors[i], continous_predictors[j]))
    # print("combinations")
    # print(combinations)

    l = []

    for pair in combinations:
        df1 = df[pair[0]]
        df2 = df[pair[1]]
        corr, _ = pearsonr(df1, df2)

        score = {
            "Predictor 1": pair[0],
            "Predictor 2": pair[1],
            "Pearson's Correlation value": corr,
        }

        # print(pair[0])
        # print(pair[1])
        # print(corr)
        l.append(score)

    score_pearson = pd.DataFrame(l)
    print(score_pearson)

    figure = correlation_matrix(
        score_pearson["Predictor 1"],
        score_pearson["Predictor 2"],
        score_pearson["Pearson's Correlation value"],
        "Correlation Pearson's Matrix",
    )

    """
    figure.write_html(
        file="adarsh21b-plots/" + pair[0] + "vs" + pair[1] + "plot_correlation.html", include_plotlyjs="cdn"
    )
    figure.show()
    """
    with open("adarsh_midterm.html", "a") as f:
        f.write('<h1 style="font-weight: bold;">Baseball Statistics</h1>')
        f.write(figure)

    l_cont = []
    for i in range(len(continous_predictors)):
        if check_response(df, response) == "Categorical":
            t_value, p_value, path = logistic_reg(
                df,
                response_df,
                df[continous_predictors[i]],
                continous_predictors[i],
                response,
            )
            print(t_value, p_value)
            score = {
                "Predictor 1": continous_predictors[i],
                "t_value": t_value,
                "p_value": p_value,
                "path": path,
            }
            l_cont.append(score)

        else:
            t_value, p_value, path = linear_reg(
                df,
                response_df,
                df[continous_predictors[i]],
                continous_predictors[i],
                response,
            )
            print(t_value, p_value)
            score = {
                "Predictor 1": continous_predictors[i],
                "t_value": t_value,
                "p_value": p_value,
                "path": path,
            }
            l_cont.append(score)

        score_df = pd.DataFrame(l_cont)

        rf_value = random_forest_value(df, response_df, continous_predictors)
        rf_score = {"rf score": rf_value}

    data = []

    # Iterate through the dictionary
    for predictor, score_dict in rf_score["rf score"].items():
        # Append a dictionary with the data for each predictor
        data.append({"random_forest_score": score_dict, "predictor": predictor})

    rf_score_df = pd.DataFrame(data)

    return combinations, score_df, rf_score_df, score_pearson


def mean_of_response_cat_cat(df, predictor, response_df, response):
    bins_mod = df[predictor].unique()

    # Calculate total population and boolean true population for each bin
    total_population_bins_mod = df.groupby(predictor)[response].agg(["count", "sum"])

    # Calculate response rate for each bin
    population_response = (
        total_population_bins_mod["sum"] / total_population_bins_mod["count"]
    )

    # Calculate true response rate for entire population
    true_value_response_rate = response_df.mean()

    # Create array of true response rate for entire population to build plot
    true_class_response_rate_population_arr = np.full(
        len(bins_mod), true_value_response_rate
    )

    # Calculate mean squared difference
    mean_squared_diff = (
        (population_response - true_class_response_rate_population_arr) ** 2
    ).mean()

    # Calculate weighted mean squared difference
    bin_population_ratio = (
        total_population_bins_mod["count"] / total_population_bins_mod["count"].sum()
    )
    weighted_mean_squared_diff = np.average(
        (population_response - true_class_response_rate_population_arr) ** 2,
        weights=bin_population_ratio,
    )

    fig = go.Figure(
        data=go.Bar(
            x=bins_mod,
            y=total_population_bins_mod,
            marker_color="#EB89B5",
            opacity=0.75,
            yaxis="y1",
        )
    )

    # scatter plot for mean of response for a class within each bin
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=population_response,
            yaxis="y2",
            name="Response",
            mode="lines+markers",
            marker=dict(color="red", size=8),
        )
    )

    # scatter plot for mean of response for the "true" value in entire population
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=true_class_response_rate_population_arr,
            yaxis="y2",
            mode="lines",
            line=dict(color="green", width=2),
            name="Overall Population Response",
        )
    )

    fig.update_layout(
        title_text=predictor + " vs " + response + " mean of response plot",
        legend=dict(orientation="v"),
        yaxis=dict(title=dict(text="Frequency"), side="left", tickformat=","),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")
    #

    # fig.show()

    file = f"adarsh21b-plots/{predictor}vs{response}morp.html"
    fig.write_html(file=file, include_plotlyjs="cdn")
    final_dict = {
        "predictor": predictor,
        "mean_squared_diff": mean_squared_diff,
        "weighted_mean_squared_diff": weighted_mean_squared_diff,
        "path": file,
    }
    return final_dict


def mean_of_response_cont_cont(df, predictor, response_df, response):
    print("Inside Mean of Response Cont-Cont")
    bin_count = 10
    bins = pd.cut(df[predictor], bin_count).apply(lambda x: x.mid)
    bins_mod = np.sort(np.array(bins.unique()))

    result = response_df.groupby(bins).agg(["sum", "count"])
    bin_population, bin_sum = np.array(result["count"]), np.array(result["sum"])
    total_population, total = np.sum(result["count"]), np.sum(result["sum"])

    bin_response = bin_sum / bin_population
    population_resp = total / total_population

    mean_squared_diff = np.nanmean((bin_response - population_resp) ** 2)
    weighted_mean_squared_diff = np.nansum(
        (bin_population / total_population) * (bin_response - population_resp) ** 2
    )

    # print("---msd---")
    # print(mean_squared_diff)
    # print("-----wmsd----")
    # print(weighted_mean_squared_diff)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bins_mod,
            y=bin_population,
            name=predictor,
            marker_color="#EB89B5",
            opacity=0.65,
            yaxis="y1",
        )
    )

    # create scatter plot for bin response
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=bin_response,
            name="Response",
            mode="lines+markers",
            marker=dict(color="red", size=8),
            yaxis="y2",
        )
    )

    # create scatter plot for population response
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=np.full_like(bins_mod, population_resp),
            name="Population Total Response",
            mode="lines",
            line=dict(color="green", width=2),
            yaxis="y2",
        )
    )

    # update layout
    fig.update_layout(
        title_text=predictor + " vs " + response + " mean of response plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Frequency"),
            side="left",
            rangemode="nonnegative",
            tickformat=",",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            rangemode="nonnegative",
            tickmode="auto",
        ),
        xaxis=dict(title=dict(text="Predictor Bins")),
    )

    # fig.show()

    file = f"adarsh21b-plots/{predictor}vs{response}morp.html"
    fig.write_html(file=file, include_plotlyjs="cdn")

    final_dict = {
        "predictor": predictor,
        "mean_squared_diff": mean_squared_diff,
        "weighted_mean_squared_diff": weighted_mean_squared_diff,
        "path_msd": file,
    }

    return final_dict


def correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """

    f_cat, _ = pandas.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def categorical_conti(df, categorical_predictors, continous_pred):
    combinations = []
    for cat_pred in categorical_predictors:
        for cont_pred in continous_pred:
            combinations.append((cat_pred, cont_pred))

    print("------------cate cont---------")
    print(combinations)

    if len(combinations) > 0:
        l_cat_cont = []
        for pair in combinations:
            df1 = df[pair[0]]
            df2 = df[pair[1]]

            ratio = correlation_ratio(df1, df2)

            score = {
                "Predictor 1": pair[0],
                "Predictor 2": pair[1],
                "Correlation value": ratio,
                "Abs Correlation value": abs(ratio),
            }
            print(pair[0])
            print(pair[1])
            print(ratio)
            l_cat_cont.append(score)

        score_df = pd.DataFrame(l_cat_cont)

        figure = correlation_matrix(
            score_df["Predictor 1"],
            score_df["Predictor 2"],
            score_df["Correlation value"],
            "Correlation Ratio",
        )

        # figure.show()

        """
        file = f"adarsh21b-plots/{pair[0]}vs{pair[1]}morp.html"
        figure.write_html(
            file=file, include_plotlyjs="cdn"
        )
        """

        with open("adarsh_midterm.html", "a") as f:
            f.write(figure)
    return score_df, combinations


def mean_of_response_cont_cat(df, predictor, response):
    bin_count = 10

    # Create the array of predictors for the overall population
    pred_vals_total = df[predictor].values

    # Use np.histogram function to get the bins and population in each bin (for bar plot)
    population, bins = np.histogram(pred_vals_total, bins=bin_count)

    # Take the average of bins to get the midpoints
    bins_mod = (bins[:-1] + bins[1:]) / 2

    # Get the mean response rate for each bin
    df["bin"] = pd.cut(df[predictor], bins=bins, labels=bins_mod)
    response_rate = df.groupby("bin")[response].mean().values

    # Create the array of "true" class mean response for entire population to build the plot
    true_class_response_rate_population_arr = np.array(
        [df[response].mean()] * bin_count
    )

    # Calculate mean squared difference
    mean_squared_diff = np.mean(
        (response_rate - true_class_response_rate_population_arr) ** 2
    )

    mean_response = df.groupby("bin")[response].mean()
    bin_population = df.groupby("bin")[response].count()

    weighted_squared_diff = np.average(
        (mean_response - true_class_response_rate_population_arr) ** 2,
        weights=bin_population,
    )

    # Bar plot for overall population
    fig = go.Figure(
        data=go.Bar(x=bins_mod, y=population, name=predictor, marker=dict(color="blue"))
    )

    # Scatter plot for mean of response for a class within each bin
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=response_rate,
            yaxis="y2",
            name="response",
            marker=dict(color="red"),
        )
    )

    # Scatter plot for mean of response for the "true" value in entire population
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=true_class_response_rate_population_arr,
            yaxis="y2",
            mode="lines",
            name="total population",
        )
    )

    fig.update_layout(
        title_text=predictor + " vs " + response + "mean of response plot",
        legend=dict(orientation="v"),
        yaxis=dict(title=dict(text="Frequency"), side="left"),
        yaxis2=dict(
            title=dict(text="Response"), side="right", overlaying="y", tickmode="auto"
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")
    # fig.show()
    file = f"adarsh21b-plots/{predictor}vs{response}morp.html"
    fig.write_html(file=file, include_plotlyjs="cdn")

    final_dict = {
        "predictor": predictor,
        "mean_squared_diff": mean_squared_diff,
        "weighted_mean_squared_diff": weighted_squared_diff,
        "path_msd": file,
    }

    return final_dict


def mean_of_response_cat_cont(df, predictor, response):
    bins = df[predictor].unique()

    bin_pop = df.groupby(predictor).size().values
    bin_sum = df.groupby(predictor)[response].sum().values
    bin_resp = bin_sum / bin_pop

    pop_resp = df[response].mean()

    mse = ((bin_resp - pop_resp) ** 2).mean()

    weighted_mse = np.average((bin_resp - pop_resp) ** 2, weights=bin_pop)

    print("---msd---")
    print(response)
    print(predictor)
    print(mse)
    print("-----wmsd----")
    print(weighted_mse)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bins,
            y=bin_pop,
            name=predictor,
            marker_color="#EB89B5",
            opacity=0.75,
            yaxis="y1",
        )
    )

    # create scatter plot for bin response
    fig.add_trace(
        go.Scatter(
            x=bins,
            y=bin_resp,
            name="Response",
            mode="lines+markers",
            marker=dict(color="red", size=8),
            yaxis="y2",
        )
    )

    # create scatter plot for population response
    fig.add_trace(
        go.Scatter(
            x=bins,
            y=np.full_like(bins, pop_resp),
            name="Population Total Response",
            mode="lines",
            line=dict(color="green", width=2),
            yaxis="y2",
        )
    )

    # update layout
    fig.update_layout(
        title_text=predictor + " vs " + response + " mean of response plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Frequency"),
            side="left",
            rangemode="nonnegative",
            tickformat=",",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            rangemode="nonnegative",
            tickmode="auto",
        ),
        xaxis=dict(title=dict(text="Predictor Bins")),
    )

    # fig.show()
    file = f"adarsh21b-plots/{predictor}vs{response}morp.html"
    fig.write_html(file=file, include_plotlyjs="cdn")

    final_dict = {
        "predictor": predictor,
        "mean_squared_diff": mse,
        "weighted_mean_squared_diff": weighted_mse,
        "path": file,
    }

    return final_dict


def brute_force_cont_cont_mean_heatmap(df, combinations, response):
    print("Inside cont cont brute force ")
    scores = []
    combinations = [(x, y) for (x, y) in combinations if x != y]
    print("combinations")
    for i in range(len(combinations)):
        pred_1, pred_2 = combinations[i]

        bins = 10
        df_new = df.copy()
        df_new["bins1"] = pd.cut(df_new[pred_1], bins=bins, right=True).apply(
            lambda x: x.mid
        )
        df_new["bins2"] = pd.cut(df_new[pred_2], bins=bins, right=True).apply(
            lambda x: x.mid
        )
        df_new = df_new[[pred_1, pred_2, "bins1", "bins2", response]]
        df_new = df_new.groupby(["bins1", "bins2"]).agg(["mean", "size"]).reset_index()
        df_new.columns = df_new.columns.to_flat_index().map("".join)

        print(df_new.columns)
        df_new["unweighted"] = (
            df_new[response + "mean"]
            .to_frame()
            .apply(lambda x: (df[response].mean() - x) ** 2)
        )
        df_new["weighted"] = df_new.apply(
            lambda x: (x[response + "size"] / df_new[response + "size"].sum())
            * x["unweighted"],
            axis=1,
        )
        df_new["meansize"] = df_new.apply(
            lambda x: "{:.3f} (pop:{})".format(
                x[response + "mean"], x[response + "size"]
            ),
            axis=1,
        )

        # print(df_new)
        # print(df_new.columns)

        weighted = df_new["weighted"].sum()
        unweighted = df_new["unweighted"].sum() / len(df_new)

        heatmap = go.Heatmap(
            x=np.array(df_new["bins1"]),
            y=np.array(df_new["bins2"]),
            z=np.array(df_new[response + "mean"]),
            text=np.array(df_new["meansize"]),
            colorscale="Blues",
            texttemplate="%{text}",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title=f"{pred_1} vs {pred_2} (Bin Averages)",
            xaxis=dict(title=pred_1),
            yaxis=dict(title=pred_2),
        )

        fig = go.Figure(data=[heatmap], layout=layout)
        file = f"adarsh21b-plots/{pred_1}vs{pred_2}heatmap.html"
        fig.write_html(file=file, include_plotlyjs="cdn")

        final_dict = {
            "Predictor 1": pred_1,
            "Predictor 2": pred_2,
            "diff_mean_resp_ranking": unweighted,
            "diff_mean_resp_weighted_ranking": weighted,
            "path": file,
        }
        scores.append(final_dict)

    mean_df = pd.DataFrame(scores)
    return mean_df


def brute_force_cat_cont_mean_heatmap(df, combinations, response):
    print(" Inside cat_cont brute force ")
    scores = []
    combinations = [(x, y) for (x, y) in combinations if x != y]
    print("combinations")
    print(combinations)

    for i in range(len(combinations)):
        cat_pred = combinations[i][0]
        cont_pred = combinations[i][1]

        print(cat_pred)
        print("===================")
        print(cont_pred)

        newdf = df
        newdf[cont_pred + "bins"] = pd.cut(df[cont_pred], bins=10, right=True).apply(
            lambda x: x.mid
        )
        newdf = (
            newdf[[cat_pred, cont_pred, cont_pred + "bins", response]]
            .groupby([cat_pred, cont_pred + "bins"])
            .agg(["mean", "size"])
            .reset_index()
        )
        newdf.columns = newdf.columns.to_flat_index().map("".join)

        newdf["unweighted"] = (
            newdf[response + "mean"]
            .to_frame()
            .apply(lambda x: (df[response].mean() - x) ** 2)
        )

        newdf["weighted"] = newdf.apply(
            lambda x: (x[response + "size"] / newdf[response + "size"].sum())
            * x["unweighted"],
            axis=1,
        )
        newdf["size"] = newdf.apply(
            lambda x: "{:.3f} (pop:{})".format(
                x[response + "mean"], x[response + "size"]
            ),
            axis=1,
        )

        unweighted = round(newdf["unweighted"].sum() / len(newdf), 6)
        weighted = round(newdf["weighted"].sum(), 6)

        # print("unweighted-------------------")
        # print(unweighted)
        # print("weighted-----------------------")
        # print(weighted)

        # # Create the heatmap trace
        heatmap = go.Heatmap(
            x=np.array(newdf[cat_pred]),
            y=np.array(newdf[cont_pred + "bins"]),
            z=np.array(newdf[response + "mean"]),
            text=np.array(newdf["size"]),
            colorscale="Blues",
            texttemplate="%{text}",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title=f"{cat_pred} vs {cont_pred} (Bin Averages)",
            xaxis=dict(title=cat_pred),
            yaxis=dict(title=cont_pred),
        )

        fig = go.Figure(data=[heatmap], layout=layout)
        file = f"adarsh21b-plots/{cat_pred}vs{cont_pred}heatmap.html"
        fig.write_html(file=file, include_plotlyjs="cdn")

        final_dict = {
            "Predictor 1": cat_pred,
            "Predictor 2": cont_pred,
            "diff_mean_resp_ranking": unweighted,
            "diff_mean_resp_weighted_ranking": weighted,
            "path": file,
        }
        scores.append(final_dict)
    mean_df = pd.DataFrame(scores)

    return mean_df


def brute_force_cat_cat_mean_heatmap(df, combinations, response):
    scores = []
    print("Inside cat cat brute force")
    combinations = [(x, y) for (x, y) in combinations if x != y]
    for i in range(len(combinations)):
        cat_pred_1 = combinations[i][0]
        cat_pred_2 = combinations[i][1]

        df_temp = (
            df.groupby([cat_pred_1, cat_pred_2])[response]
            .agg(["mean", "size"])
            .reset_index()
        )
        df_temp.columns = [cat_pred_1, cat_pred_2, "mean", "size"]

        df_temp["unweighted"] = (df_temp["mean"].mean() - df_temp["mean"]) ** 2
        df_temp["weighted"] = df_temp["unweighted"] * (
            df_temp["size"] / df_temp["size"].sum()
        )

        unweighted = df_temp["unweighted"].sum() / (
            df[cat_pred_1].nunique() * df[cat_pred_2].nunique()
        )
        weighted = df_temp["weighted"].sum()

        print("weighted------------")
        print(weighted)
        print("unweighted-------------")
        print(unweighted)

        df_temp["mean_size"] = (
            df_temp["mean"].round(3).astype(str)
            + " (pop: "
            + df_temp["size"].astype(str)
            + ")"
        )

        # Create the heatmap trace
        heatmap = go.Heatmap(
            x=np.array(df_temp[cat_pred_1]),
            y=np.array(df_temp[cat_pred_2]),
            z=np.array(df_temp["mean"]),
            text=np.array(df_temp["mean_size"]),
            colorscale="Blues",
            texttemplate="%{text}",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title=f"{cat_pred_1} vs {cat_pred_2} (Bin Averages)",
            xaxis=dict(title=cat_pred_1),
            yaxis=dict(title=cat_pred_2),
        )

        fig = go.Figure(data=[heatmap], layout=layout)
        file = f"adarsh21b-plots/{cat_pred_1}vs{cat_pred_2}heatmap.html"
        fig.write_html(file=file, include_plotlyjs="cdn")

        # fig.show()
        final_dict = {
            "Predictor 1": cat_pred_1,
            "Predictor 2": cat_pred_2,
            "diff_mean_resp_ranking": unweighted,
            "diff_mean_resp_weighted_ranking": weighted,
            "path": file,
        }
        scores.append(final_dict)
        mean_df = pd.DataFrame(scores)

    return mean_df


def categorical_cate(df, categorical_predictors, response_df):
    combinations = []
    for i in range(len(categorical_predictors)):
        for j in range(len(categorical_predictors)):
            # if categorical_predictors[i] != categorical_predictors[j]:
            combinations.append((categorical_predictors[i], categorical_predictors[j]))
    print("combinations")
    print(combinations)

    l_t = []
    l_c = []

    for pair in combinations:
        df1 = df[pair[0]]
        df2 = df[pair[1]]

        correlation = cat_correlation(df1, df2)
        score_c = {
            "Predictor 1": pair[0],
            "Predictor 2": pair[1],
            "Cramer's V": correlation,
            "Correlation value": abs(correlation),
        }
        l_c.append(score_c)

        score_df_c = pd.DataFrame(l_c)

        correlation = cat_correlation(df1, df2, tschuprow=True)

        score = {
            "Predictor 1": pair[0],
            "Predictor 2": pair[1],
            "tschuprow": correlation,
            "Correlation value": abs(correlation),
        }

        l_t.append(score)
        score_df = pd.DataFrame(l_t)

    figure_c = correlation_matrix(
        score_df_c["Predictor 1"],
        score_df_c["Predictor 2"],
        score_df_c["Correlation value"],
        "Cramer's V Heatmap",
    )
    with open("adarsh_midterm.html", "a") as f:
        f.write(figure_c)

    # figure_c.show()
    """
    file = f"adarsh21b-plots/{pair[0]}vs{pair[1]}cramer.html"
    figure_c.write_html(
        file=file, include_plotlyjs="cdn"
    )
    """
    # fig = px.density_heatmap(x=df1, y=df2, color_continuous_scale="Viridis", text_auto=True)
    # title = f" categorical predictor ({df1.name}) by categorical response ({df2.name})"
    # fig.update_layout(
    #     title=title,
    #     xaxis_title=f"Predictor ({df1.name})",
    #     yaxis_title=f"Predictor ({df2.name})",
    # )
    # fig.show()

    figure = correlation_matrix(
        score_df["Predictor 1"],
        score_df["Predictor 2"],
        score_df["Correlation value"],
        "Tschuprow Heatmap",
    )

    # figure.show()

    # file = f"adarsh21b-plots/{pair[0]}vs{pair[1]}tschuprow.html"

    """
    figure.write_html(
        file=file, include_plotlyjs="cdn"
    )
    """
    with open("adarsh_midterm.html", "a") as f:
        f.write(figure)
    return combinations, score_df, score_df_c


def cont_response_cat_predictor_plot(df, predictor, response):
    group_labels = df[predictor].unique()
    hist_data = []

    for label in group_labels:
        hist_data.append(np.array(df[df[predictor] == label][response]))

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, len(curr_group)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title=f"Continuous Response {response} by Categorical Predictor {predictor}",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    # fig_2.show()

    file = f"adarsh21b-plots/{predictor}plot.html"
    fig_2.write_html(file=file, include_plotlyjs="cdn")

    summary_dict = {"predictor": predictor, "path": file}
    return summary_dict


def merge_cat_cat_correlation_dataframes_c(df_cramers, df_mean_cat, df_mean_cont):
    # Creating cramers merged dataframe

    # merge the two dataframes on the 'Pred1' column of the first dataframe and the 'Pred' column of the second

    merged_df = pd.merge(
        df_cramers,
        df_mean_cat,
        left_on="Predictor 1",
        right_on="predictor",
    ).drop("predictor", axis=1)

    # merge the resulting dataframe with the second dataframe again, this time on the 'Pred2' column of the first df

    merged_df = pd.merge(
        merged_df,
        df_mean_cat,
        left_on="Predictor 2",
        right_on="predictor",
        suffixes=["1", "2"],
    ).drop("predictor", axis=1)

    merged_df = merged_df[
        ["Predictor 1", "Predictor 2", "Correlation value", "path1", "path2"]
    ]
    save_dataframe_to_html(
        merged_df,
        "path1",
        "path2",
        "Cramer's Correlation",
    )


def merge_cat_cat_correlation_dataframes_t(df_tschuprow, df_mean_cat, df_mean_cont):

    merged_df = pd.merge(
        df_tschuprow, df_mean_cat, left_on="Predictor 1", right_on="predictor"
    ).drop("predictor", axis=1)

    # merge the resulting dataframe with the second dataframe again
    merged_df = pd.merge(
        merged_df,
        df_mean_cat,
        left_on="Predictor 2",
        right_on="predictor",
        suffixes=["1", "2"],
    ).drop("predictor", axis=1)

    merged_df = merged_df[
        ["Predictor 1", "Predictor 2", "Correlation value", "path1", "path2"]
    ]

    save_dataframe_to_html(
        merged_df,
        "path1",
        "path2",
        "Tschuprow Correlation",
    )


def save_dataframe_to_html_bruteforce(df, link, caption):
    def make_clickable(link, link_name):
        # Create an HTML link tag with the link and link_name
        return f'<a href="{link}">{link_name}</a>'

    # Apply styles to the DataFrame using the Styler class
    styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {
            "selector": "th",
            "props": [
                ("background-color", "#88bde6"),
                ("text-align", "center"),
                ("color", "white"),
            ],
        },
        {
            "selector": "th, td",
            "props": [("padding", "10px"), ("border", "2px solid #c4d4e3")],
        },
        {"selector": "tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
        {"selector": "tr:hover", "props": [("background-color", "#e6f2ff")]},
    ]
    # Format the columns with clickable links
    df_styled = df.style.format({link: make_clickable})

    # Apply table styles and add a caption
    df_styled.set_table_styles(styles).set_caption(
        f"<h1 style='text-align:center; font-size:30px; color:#88bde6;'>{caption}</h1>"
    )

    # Format the columns with clickable links
    df_styled = df.style.format({link: lambda x: make_clickable(x, "Plot Link")})

    # Apply table styles and add a caption
    df_styled.set_table_styles(styles).set_caption(
        f"<h2 style='text-align:center; font-size:25px;'>{caption}</h1>"
    )
    # Generate an HTML table from the styled DataFrame
    html_table = df_styled.to_html()

    with open("adarsh_midterm.html", "a") as f:
        f.write(html_table)


def save_dataframe_to_html(df, plot_link_mor, plot_link, caption):
    def make_clickable(link, link_name):
        # Create an HTML link tag with the link and link_name
        return f'<a href="{link}">{link_name}</a>'

    # Apply styles to the DataFrame using the Styler class
    styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {
            "selector": "th",
            "props": [
                ("background-color", "#88bde6"),
                ("text-align", "center"),
                ("color", "white"),
            ],
        },
        {
            "selector": "th, td",
            "props": [("padding", "10px"), ("border", "2px solid #c4d4e3")],
        },
        {"selector": "tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
        {"selector": "tr:hover", "props": [("background-color", "#e6f2ff")]},
    ]
    # Format the columns with clickable links
    df_styled = df.style.format(
        {plot_link_mor: make_clickable, plot_link: make_clickable}
    )

    # Apply table styles and add a caption
    df_styled.set_table_styles(styles).set_caption(
        f"<h1 style='text-align:center; font-size:30px; color:#88bde6;'>{caption}</h1>"
    )

    df_styled = df.style.format(
        {
            plot_link_mor: lambda x: make_clickable(x, "Plot Link Mor"),
            plot_link: lambda x: make_clickable(x, "Plot Link"),
        }
    )

    # Apply table styles and add a caption
    df_styled.set_table_styles(styles).set_caption(
        f"<h2 style='text-align:center; font-size:25px;'>{caption}</h1>"
    )
    # Generate an HTML table from the styled DataFrame
    html_table = df_styled.to_html()

    with open("adarsh_midterm.html", "a") as f:
        f.write(html_table)


def get_baseball_data():
    db_user = "root"
    db_pass = 9501
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
        SELECT * FROM final_baseball
    """
    df = pd.read_sql_query(query, sql_engine)
    return df


def build_model(df, predictors, response):
    X = df[[predictors]].values
    y = df[response].values

    # Train and split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, stratify=None, random_state=2
    )

    # Decision Tree Model
    clf2 = tree.DecisionTreeClassifier()
    clf2 = clf2.fit(X_train, y_train)

    print("Decision Tree : ", clf2.score(X_test, y_test))

    # Random Forest Model
    pipeline = Pipeline(
        [
            ("OneHotEncode", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=100)),
        ]
    )
    pipeline.fit(X_train, y_train)
    Pipeline(
        steps=[
            ("OneHotEncode", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=100)),
        ]
    )
    print("Random Forest Score:", pipeline.score(X_test, y_test))

    # Logistic Regression Model
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    print("Logistic Regression Score : ", clf.score(X_test, y_test))

    print(
        "Almost similar scores are observed in these three above models. Overall I would say by comparing the "
        "scores RANDOM FOREST would be good algorithm to use"
    )


def save_dataframe_to_html_pearson(df, plot_link, caption):
    def make_clickable(link, link_name):
        # Create an HTML link tag with the link and link_name
        return f'<a href="{link}">{link_name}</a>'

    # Apply styles to the DataFrame using the Styler class
    styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {
            "selector": "th",
            "props": [
                ("background-color", "#88bde6"),
                ("text-align", "center"),
                ("color", "white"),
            ],
        },
        {
            "selector": "th, td",
            "props": [("padding", "10px"), ("border", "2px solid #c4d4e3")],
        },
        {"selector": "tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
        {"selector": "tr:hover", "props": [("background-color", "#e6f2ff")]},
    ]

    df_styled = df.style.format({plot_link: lambda x: make_clickable(x, "Plot Link")})

    # Apply table styles and add a caption
    df_styled.set_table_styles(styles).set_caption(
        f"<h1 style='text-align:center; font-size:30px; color:#88bde6;'>{caption}</h1>"
    )

    # Apply table styles and add a caption
    df_styled.set_table_styles(styles).set_caption(
        f"<h2 style='text-align:center; font-size:25px;'>{caption}</h1>"
    )
    # Generate an HTML table from the styled DataFrame
    html_table = df_styled.to_html()

    with open("adarsh_midterm.html", "a") as f:
        f.write(html_table)


def main():
    check_directory()
    df = get_baseball_data()
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    response = "Home_Team_Wins"
    predictors = [
        "Isolated_Power_Diff",
        "DICE_diff",
        "Whip_100_diff",
        "Home_Run_per_Nine_Innings_diff",
        "Hits_per_Nine_Innings_diff",
        "Batting_Avg_Against_diff",
        "CERA_diff",
        "PFR_diff",
        "TimesOnBaseDiff",
        "StrikeOutToWalkRatio_diff",
    ]
    print("dataframe")
    print(df)
    print("predictors")
    print(predictors)
    print("response")
    print(response)

    print("Response type: " + str(check_response(df, response)))
    response_df = df[response]

    continuous_predictors = []
    for predictor in predictors:
        if check_predictor(df, predictor) == "Continuous":
            continuous_predictors.append(predictor)
    print(continuous_predictors)

    categorical_predictors = []
    for predictor in predictors:
        print("check categorical predictor")
        if check_predictor(df, predictor) == "Categorical":
            categorical_predictors.append(predictor)

    print("----------------categorical predictors-----------")
    print(categorical_predictors)

    # continous_continous
    cont_df = pd.DataFrame()
    if len(continuous_predictors) >= 1:
        cont_df, score_df, rf_score_df, pearson_df = continuous_cont(
            df, continuous_predictors, response_df, response
        )

        cont_df_temporary = score_df.merge(
            rf_score_df, left_on="Predictor 1", right_on="predictor"
        )
        cont_df_temporary = cont_df_temporary.drop("predictor", axis=1)

    l1 = []

    for i in range(len(continuous_predictors)):
        pred = continuous_predictors[i]
        if check_response(df, response) == "Continuous":
            mean_dict = mean_of_response_cont_cont(df, pred, response_df, response)
            l1.append(mean_dict)
        else:
            mean_dict = mean_of_response_cont_cat(df, pred, response)
            l1.append(mean_dict)

    mean_df_cont_pred = pd.DataFrame(l1)
    merged_df = pd.merge(
        mean_df_cont_pred, pearson_df, left_on="predictor", right_on="Predictor 1"
    )
    merged_df = merged_df.drop("Predictor 1", axis=1)
    merged_df = merged_df.rename(columns={"Predictor 2": "predictor_2"})
    merged_df = merged_df[
        ["predictor", "predictor_2", "Pearson's Correlation value", "path_msd"]
    ]
    save_dataframe_to_html_pearson(merged_df, "path_msd", "Correlation Pearson's Table")

    temp_merged = cont_df_temporary.merge(
        mean_df_cont_pred, left_on="Predictor 1", right_on="predictor"
    )
    temp_merged = temp_merged.drop("predictor", axis=1)

    save_dataframe_to_html(
        temp_merged,
        "path",
        "path_msd",
        "Continuous Predictors",
    )

    l2 = []
    cate_df = pd.DataFrame()
    if len(categorical_predictors) >= 1:
        cate_df, t_df, cramer_df = categorical_cate(
            df, categorical_predictors, response_df
        )

    for i in range(len(categorical_predictors)):
        predictor = categorical_predictors[i]
        if check_response(df, response) == "Categorical":
            mean_dict = mean_of_response_cat_cat(df, predictor, response_df, response)
            l2.append(mean_dict)
        else:
            # summary_dict = cont_response_cat_predictor_plot(df, predictor, response)
            mean_dict = mean_of_response_cat_cont(df, predictor, response)
            l2.append(mean_dict)

    mean_df_cat_pred = pd.DataFrame(l2)
    score_df_corr = pd.DataFrame()
    if len(categorical_predictors) > 0 and len(continuous_predictors) > 0:
        score_df_corr, combinations_cat_cont = categorical_conti(
            df, categorical_predictors, continuous_predictors
        )

    # Merging the dataframes for cramer's table
    if len(categorical_predictors) > 0:
        merge_cat_cat_correlation_dataframes_c(
            cramer_df, mean_df_cat_pred, mean_df_cont_pred
        )
        merge_cat_cat_correlation_dataframes_t(
            t_df, mean_df_cat_pred, mean_df_cont_pred
        )

        # Brute Force
        brute_force_mean_df = brute_force_cat_cat_mean_heatmap(df, cate_df, response)
        merged_df = pd.merge(
            pd.merge(brute_force_mean_df, cramer_df, on=["Predictor 1", "Predictor 2"]),
            t_df,
            on=["Predictor 1", "Predictor 2"],
        )

        save_dataframe_to_html_bruteforce(
            merged_df,
            "path",
            "Brute_Force_Categorical_Categorical",
        )

    if len(continuous_predictors) > 0:
        brute_force_cont_cont_df = brute_force_cont_cont_mean_heatmap(
            df, cont_df, response
        )
        merged_df = pd.merge(
            brute_force_cont_cont_df, pearson_df, on=["Predictor 1", "Predictor 2"]
        )
        save_dataframe_to_html_bruteforce(
            merged_df,
            "path",
            "Brute_Force_Continuous_Continuous",
        )

    if len(continuous_predictors) > 0 and len(categorical_predictors) > 0:
        brute_force_cat_cont_df = brute_force_cat_cont_mean_heatmap(
            df, combinations_cat_cont, response
        )

        merged_df = pd.merge(
            brute_force_cat_cont_df, score_df_corr, on=["Predictor 1", "Predictor 2"]
        )
        save_dataframe_to_html_bruteforce(
            merged_df,
            "path",
            "Brute_Force_Categorical_Continuous",
        )

    # Training model and features

    print(predictors)
    build_model(df, predictor, response)


if __name__ == "__main__":
    sys.exit(main())
