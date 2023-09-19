import os
import time

#from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import vineyard


def preprocess_data():
    os.system('sync; echo 3 > /proc/sys/vm/drop_caches')
    data_multiplier = os.environ.get('DATA_MULTIPLIER', 1)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    del X, y

    st = time.time()
    with_vineyard = os.environ.get('WITH_VINEYARD', False)
    if with_vineyard:
        vineyard.csi.write(X_train, "/data/x_train.pkl")
        vineyard.csi.write(X_test, "/data/x_test.pkl")
        vineyard.csi.write(y_train, "/data/y_train.pkl")
        vineyard.csi.write(y_test, "/data/y_test.pkl")
    else:
        X_train.to_pickle('/data/x_train.pkl')
        X_test.to_pickle('/data/x_test.pkl')
        y_train.to_pickle('/data/y_train.pkl')
        y_test.to_pickle('/data/y_test.pkl')

    ed = time.time()
    print('##################################')
    print('write training and testing data time: ', ed - st)


if __name__ == '__main__':
    st = time.time()
    print('Preprocessing data...')
    preprocess_data()
    ed = time.time()
    print('##################################')
    print('Preprocessing data time: ', ed - st)
