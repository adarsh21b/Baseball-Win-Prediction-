# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update dev requirements: `pip-compile --output-file=requirements.dev.txt requirements.dev.in`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# BASEBALL WIN PREDICTION USING MACHINE LEARNING

## Introduction
Baseball Dataset: The dataset contains Batting, Pitching and Fielding statistics of team. There are numerous tables in this database that contain information about baseball players and teams, including game, box score, team batting, team pitching, and pitcher counts, among others. 
Dataset Link: 

(This database provides data from baseball contests that took place between 2008 and 2012.)

- The goal of this research is to use the baseball dataset to develop new features that may be used to forecast whether the home team will win a specific game or not. We will be doing feature engineering in order to do this.
- This kind of project could be used in a variety of ways, such as assisting sports analysts and broadcasters in making more precise predictions on-air, assisting fans in making more informed choices about which games to watch or attend, offering teams and players advice on possible strategies and tactics, and more.
- Target Variable (Response) : Home Team Wins
- Home Team Wins, the project's primary outcome measure, was created by subtracting the scores earned by the home team from those of the away team in the box_score database.
- In order to understand this numerous plots, scores, metrics and correlation ratio is done along with machine learning models to predict whether the home team wins or not ?
- By observing how different predictors interact with one another, these plots help us to discern between predictors that are helpful for our prediction analysis and those that are not.
- The ultimate objective would be to gradually increase the accuracy of these forecasts, maybe by adding more data or modifying the model in response to its performance.
- Brute force approach is also done to understand the features importance along with its mean of response plot to improve the understanding.
- Which is the best ML model so far ? ( We will use different ML Algorithms like Random Forest, Logisitc Regression, SVM and other SUpervised Learning alogorithms to evaluate model performance)
- Metrics like recall, accuracy, and precision are used to gauge the model's performance.

## Tools and Libraries Used
- [Python](https://www.python.org/downloads/) - Numpy, Pandas, sqlalchemy, sklearn, plotly, statsmodel, plotly, scipy etc.
- [Docker](https://www.docker.com/)
- [Mariadb](https://mariadb.com/kb/en/getting-installing-and-upgrading-mariadb/)
- [SQL](https://mariadb.com/kb/en/sql-statements/) 

## Features and its Importance
Feature engineering is a machine learning method that creates new variables not found in the training set by using data. It can provide new features for both supervised and unsupervised learning, with the goal of streamlining and speeding up data conversions while simultaneously enhancing model correctness. The time-effective method of feature engineering for getting consistent results from data preparation makes its significance clear. An unfavorable feature will have a direct impact on your model, regardless of the architecture or the data.

# Features Used in Baseball Dataset
1. [Isolated power](https://en.wikipedia.org/wiki/Isolated_power)
2. [DICE (Defense Component ERA)](https://en.wikipedia.org/wiki/Defense-Independent_Component_ERA)
3. [WHIP (Walks and Hits Per Innings)](https://en.wikipedia.org/wiki/Walks_plus_hits_per_inning_pitched)
4. [Hits allowed per nine Innings](https://en.wikipedia.org/wiki/Hits_per_nine_innings)
5. [Strikeout per nine innings pitched](https://en.wikipedia.org/wiki/Strikeouts_per_nine_innings_pitched)
6. [Base on Balls per nine innings](https://en.wikipedia.org/wiki/Bases_on_balls_per_nine_innings_pitched)
7. [Home runs per nine innings](https://en.wikipedia.org/wiki/Home_runs_per_nine_innings)
8. [CERA (Component ERA)](https://en.wikipedia.org/wiki/Component_ERA)
9. [Times on Base](https://en.wikipedia.org/wiki/Times_on_base)
10. [Batting Avgerage_Against](https://en.wikipedia.org/wiki/Batting_average_against)
11. [PFR (Power Finesse Ratio)](https://en.wikipedia.org/wiki/Power_finesse_ratio)
12. [Strikeout to Walk Ratio](https://en.wikipedia.org/wiki/Strikeout-to-walk_ratio)
13. [Pythagorean Expectation](https://en.wikipedia.org/wiki/Pythagorean_expectation)
14. [BABIP (Batting Average on Balls in Play)](https://en.wikipedia.org/wiki/Batting_average_on_balls_in_play)

# About some Important Features
## DICE
Similar to CERA, but using a different model that takes into account additional factors including the strikeout rate and walk rate.
These factors can be very important since they show how well-controlled a pitcher is and how well-equipped they are to get batters out on their own, without the assistance of the defense.If DICE is incorporated into the model as a feature, it may be able to more accurately predict game outcomes based on the relative strength of each team's pitching staff.

## CERA 
This measure seeks to forecast a pitcher's earned run average (ERA).
This could be a key component of the model given that ERA is a common indicator for evaluating pitcher performance. It gives an indication of how many runs an average pitcher gives up throughout a game and is widely used to compare pitchers.
The relative power of each team's pitching staff could be a key factor in determining the outcome of a game, and the CERA feature could help the model grasp this better. This is accomplished by foreseeing a pitcher's ERA

## WHIP
The WHIP is one of the metrics most frequently used to evaluate a pitcher's performance. The formula is simple: divide a pitcher's total innings pitched by the total of his walks and hits. It stands to reason that the best pitchers in the league should be able to stop baserunners because they often have the lowest WHIPs. However, WHIP does not consider the means by which a hitter reached base. Naturally, home runs hurt pitchers more than walks do. Hit batsmen, errors, or fielder decisions have no impact on a pitcher's WHIP.




