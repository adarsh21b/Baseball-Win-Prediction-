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
Baseball Dataset: The dataset contains Batting, Pitching and Fielding statistics of team.  
Dataset Link: 

- The goal of this research is to use the baseball dataset to develop new features that may be used to forecast whether the home team will win a specific game or not. We will be doing feature engineering in order to do this.
- Target Variable (Response) : Home Team Wins
- In order to understand this numerous plots, scores, metrics and correlation ratio is done along with machine learning models to predict whether the home team wins or not.
- By observing how different predictors interact with one another, these plots help us to discern between predictors that are helpful for our - prediction analysis and those that are not.
- Brute force approach is also done to understand the features importance along with its mean of response plot to improve the understanding.
Which is the best ML model so far ?



