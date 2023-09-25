import os
import time

from sklearn.linear_model import LinearRegression

import joblib
import pandas as pd
import vineyard


def train_model():
    os.system('sync; echo 3 > /proc/sys/vm/drop_caches')
    st = time.time()
    with_vineyard = os.environ.get('WITH_VINEYARD', False)
    if with_vineyard:
        x_train_data = vineyard.csi.read("/vineyard/data/x_train.pkl")
        y_train_data = vineyard.csi.read("/vineyard/data/y_train.pkl")
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
    st = time.time()
    print('Training model...')
    train_model()
    ed = time.time()
    print('##################################')
    print('Training model data time: ', ed - st)
