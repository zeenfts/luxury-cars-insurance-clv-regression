# libraries
RANDOM_STATE = 233

# basic analysis & viz
import numpy as np
np.random.seed(RANDOM_STATE)

import pandas as pd
import seaborn as sns
import missingno as msn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use('ggplot')
sns.set_style('darkgrid', {'grid.background': 'blue'})

# statistics
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from statsmodels.stats.outliers_influence import variance_inflation_factor

# from sklearn.datasets import load_boston
# from fairlearn.datasets import fetch_boston

# split & metrics
from sklearn.model_selection import (
    cross_validate, cross_val_score,
    GridSearchCV, KFold, RandomizedSearchCV, 
    train_test_split, StratifiedKFold 
)
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, make_scorer,
    root_mean_squared_error, PredictionErrorDisplay
)
from sklearn.metrics._scorer import _SCORERS

# preprocessing
import smogn
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import (
    OneHotEncoder, KBinsDiscretizer, MinMaxScaler,
    OrdinalEncoder, RobustScaler, StandardScaler,
    PolynomialFeatures, 
)
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, RFECV

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from category_encoders import BinaryEncoder

# models
import shap
# shap.initjs()

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge,
    HuberRegressor, TheilSenRegressor,
    QuantileRegressor, RANSACRegressor
)

from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor,
    VotingRegressor, RandomForestRegressor
)
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor

# others
import sys
import time
import pickle

import warnings
# warnings.filterwarnings('ignore')

# ===================================================================================================

# if __name__ == '__main__':
def check_nullity_uniqueness(car_clv_insurance):
    '''
    * car_clv_insurance: The dataframe of dataset.
    '''
    display(
    pd.DataFrame({
        'feature': car_clv_insurance.columns.values,
        'data_type': car_clv_insurance.dtypes.values,
        'null': car_clv_insurance.isna().mean().values * 100,
        'negative': [
            True if car_clv_insurance[i].dtypes == int and (car_clv_insurance[i] < 0).any()
            else False for i in car_clv_insurance.columns
        ],
        'n_unique': car_clv_insurance.nunique().values,
        'sample_unique': [car_clv_insurance[i].unique() for i in car_clv_insurance.columns]
    }))

def show_uniqueness_value(car_clv_insurance):
    '''
    * car_clv_insurance: The dataframe of dataset.
    '''
    categorical_features = ['Vehicle Class', 'Coverage', 'EmploymentStatus', 'Marital Status', 'Education']
    for k in categorical_features:
        display(
            car_clv_insurance[k].value_counts()
        )

def plot_distribution_outliers(car_clv_insurance):
    # Potential Outliers on Numerical Features
    # Setting up the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(21, 15))
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    # List of features to plot
    numerical_features = [
        'Number of Policies',
        'Monthly Premium Auto',
        'Total Claim Amount',
        'Income',
        'Customer Lifetime Value'
        ]

    # Plotting each feature in the respective subplot
    for ax, feature in zip(axes.flatten(), numerical_features):
        sns.boxplot(
            x = car_clv_insurance[feature],
            ax = ax,
            orient = 'horizontal',
            color = '#358EFF'
        )
        ax.set_title(f'Distribution of `{feature}`')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Removing any unused subplots (if any)
    for i in range(len(numerical_features), 6):
        fig.delaxes(axes.flatten()[i])

    plt.show()

def calculate_outliers(data, column):
    Q1 = data[column].quantile(.25)
    Q3 = data[column].quantile(.75)
    IQR = Q3 - Q1
    # lower_bound = abs(Q1 - 1.5*IQR) * 0 # reasonable for this numeric to have mininmum value '0'
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    outliers = data[
        (data[column] < lower_bound) | (data[column] > upper_bound)
    ]
    return outliers, lower_bound, upper_bound

def scatter_numerical_vs_target(
        data, numerical_features,
        target = 'Customer Lifetime Value',
        hue='Outlier Check'
    ):
    # Setting up the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(21, 15))
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    # Plotting each feature in the respective subplot
    for ax, feature in zip(axes.flatten(), numerical_features):
        sns.scatterplot(
            data = data,
            x = feature,
            y = target,
            hue = hue,
            style = hue,
            palette = {True: 'red', False: '#358EFF'},
            ax = ax
            )
        ax.set_title(f'`{feature}` vs `{target}`')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Removing any unused subplots (if any)
    for i in range(len(numerical_features), 6):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

def stripplot_categorical_vs_target(
        data, categorical_features, 
        target = 'Customer Lifetime Value',
        hue='Outlier Check'
    ):
    # Setting up the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(21, 15))
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    # Plotting each feature in the respective subplot
    for ax, feature in zip(axes.flatten(), categorical_features):
        sns.stripplot(
            data = data,
            x = feature,
            y = target,
            hue = hue,
            # marker = ['X', 'D'],
            palette = {True: 'red', False: '#358EFF'},
            ax = ax
            )
        ax.set_title(f'`{feature}` vs `{target}`')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Removing any unused subplots (if any)
    for i in range(len(categorical_features), 6):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()