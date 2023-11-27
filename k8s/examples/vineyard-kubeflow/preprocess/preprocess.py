import argparse
import os
import time

#from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import vineyard


def preprocess_data(data_multiplier, with_vineyard):
    os.system('sync; echo 3 > /proc/sys/vm/drop_caches')
    st = time.time()
    df = pd.read_pickle('/data/df_{0}.pkl'.format(data_multiplier))

    ed = time.time()
    print('##################################')
    print('read dataframe pickle time: ', ed - st)

    df = df.drop(df[(df['GrLivArea']>4800)].index)

    """ The following part will need large memory usage, disable for benchmark
    del df

    # Define the categorical feature columns
    categorical_features = df_preocessed.select_dtypes(include='object').columns

    # Create the column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    # Preprocess the features using the column transformer
    one_hot_df = preprocessor.fit_transform(df_preocessed)

    # Get the column names for the encoded features
    encoded_feature_names = preprocessor.named_transformers_['encoder'].get_feature_names_out(categorical_features)

    columns = list(encoded_feature_names) + list(df_preocessed.select_dtypes(exclude='object').columns)

    del df_preocessed

    # Concatenate the encoded features with the original numerical features
    df = pd.DataFrame(one_hot_df, columns=columns)

    del one_hot_df
    """

    X = df.drop('SalePrice', axis=1)  # Features
    y = df['SalePrice']  # Target variable

    del df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    del X, y

    st = time.time()
    if with_vineyard:
        client = vineyard.connect()
        client.put(X_train, name="/data/x_train.pkl", persist=True)
        client.put(X_test, name="/data/x_test.pkl", persist=True)
        client.put(y_train, name="/data/y_train.pkl", persist=True)
        client.put(y_test, name="/data/y_test.pkl", persist=True)
    else:
        X_train.to_pickle('/data/x_train.pkl')
        X_test.to_pickle('/data/x_test.pkl')
        y_train.to_pickle('/data/y_train.pkl')
        y_test.to_pickle('/data/y_test.pkl')

    ed = time.time()
    print('##################################')
    print('write training and testing data time: ', ed - st)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_multiplier', type=int, default=1, help='Multiplier for data')
    parser.add_argument('--with_vineyard', type=bool, default=False, help='Whether to use vineyard')
    args = parser.parse_args()
    st = time.time()
    print('Preprocessing data...')
    preprocess_data(args.data_multiplier, args.with_vineyard)
    ed = time.time()
    print('##################################')
    print('Preprocessing data time: ', ed - st)
