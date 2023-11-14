import argparse
import os
import time

from sklearn.linear_model import LinearRegression

import joblib
import pandas as pd
import vineyard


def train_model(with_vineyard):
    os.system('sync; echo 3 > /proc/sys/vm/drop_caches')
    st = time.time()
    if with_vineyard:
        client = vineyard.connect()
        x_train_data = client.get(name="/data/x_train.pkl", fetch=True)
        y_train_data = client.get(name="/data/y_train.pkl", fetch=True)
    else:
        x_train_data = pd.read_pickle("/data/x_train.pkl")
        y_train_data = pd.read_pickle("/data/y_train.pkl")
        # delete the x_train.pkl and y_train.pkl
        os.remove("/data/x_train.pkl")
        os.remove("/data/y_train.pkl")
    ed = time.time()
    print('##################################')
    print('read x_train and y_train data time: ', ed - st)

    model = LinearRegression()
    model.fit(x_train_data, y_train_data)

    joblib.dump(model, '/data/model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_vineyard', type=bool, default=False, help='Whether to use vineyard')
    args = parser.parse_args()
    st = time.time()
    print('Training model...')
    train_model(args.with_vineyard)
    ed = time.time()
    print('##################################')
    print('Training model data time: ', ed - st)
