import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

from .utils import generate_random_dataframe

from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer


def data_augment(train: pd.DataFrame, multiplier: int = None) -> pd.DataFrame:
    if multiplier is None:
        multiplier = int(os.environ.get('DATA_AUGMENT_MULTIPLIER', '1'))
    random_dataframe = generate_random_dataframe(len(train)*(multiplier-1))
    train = pd.concat([train, random_dataframe], ignore_index=True)
    return train

def remove_outliers(train: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    train = data_augment(train)
    train = train.drop(train[(train['GrLivArea']>parameters['outliers']['GrLivArea']) & 
                             (train['SalePrice']<parameters['outliers']['SalePrice'])]
                       .index)
    return train

def create_target(train: pd.DataFrame) -> pd.DataFrame:
    y_train = np.log1p(train["SalePrice"])
    return y_train

def drop_cols(train: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    return train.drop(parameters['cols_to_drop'],axis=1)

def fill_na(train: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    train[parameters['none_cols']] = train[parameters['none_cols']].fillna("None")
    train[parameters['zero_cols']] = train[parameters['zero_cols']].fillna(0)

    train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
                                        lambda x: x.fillna(x.median()))

    impute_int = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    impute_str = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    int_cols = train.select_dtypes(include='number').columns
    str_cols = train.select_dtypes(exclude='number').columns

    train[int_cols] = impute_int.fit_transform(train[int_cols])
    train[str_cols] = impute_str.fit_transform(train[str_cols])

    return train

def total_sf(train: pd.DataFrame) -> pd.DataFrame:
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    return train
