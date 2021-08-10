#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Adapted from example dags in Airflow's documentation, see also
#
#  https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html
#

from datetime import datetime

import numpy as np
import pandas as pd
from airflow.decorators import dag, task
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'start_date': days_ago(2),
}


@task()
def build_dataframe() -> pd.DataFrame:
    """
    #### build random dataframe task
    """
    df = pd.DataFrame(np.random.randint(0, 1000, size=(1000, 6)), columns=list("ABCDEF"))

    return df


@task()
def sum_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    #### Transform
    """
    print('df = ', df)
    return pd.DataFrame(df.sum()).T


@task()
def pick_least(df: pd.DataFrame) -> int:
    min_val = df.T.min()
    print("Min Val is %d" % min_val)
    return min_val


@task()
def pick_greatest(df: pd.DataFrame) -> int:
    max_val = df.T.max()
    print("Max Val is %d" % max_val)
    return max_val


@task()
def calc_mean(df: pd.DataFrame) -> int:
    mean = df.T.mean()
    print("Mean Val is %.2f" % mean)
    return mean


@task()
def calc_std_dev(df: pd.DataFrame) -> int:
    std = df.T.std()
    print("Std Deviation is %.2f" % std)
    return std


@task()
def calc_variance(df: pd.DataFrame) -> int:
    var = df.T.var()
    print("Variance Val is %.2f" % var)
    return var


@task()
def calc_median(df: pd.DataFrame) -> int:
    median = df.T.median()
    print("Median Val is %.2f" % median)
    return median


@task()
def load_results(min_val: int, max_val: int, mean: float, std: float, variance: float, median: float) -> None:
    """
    #### Load task
    This will print max and min
    """

    print("The final min is: %d" % min_val)
    print("The final max is: %d" % max_val)
    print("The final mean is: %.2f" % mean)
    print("The final std dev is: %.2f" % std)
    print("The final var is: %.2f" % variance)
    print("The final median is: %.2f" % median)


@dag(
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2021, 3, 11),
    tags=["finished-pandas-example"],
    catchup=False,
)
def taskflow_v6d():

    build_raw_df = build_dataframe()
    sum_cols_r = sum_cols(build_raw_df)
    pick_least_r = pick_least(sum_cols_r)
    pick_greatest_r = pick_greatest(sum_cols_r)
    calc_mean_r = calc_mean(sum_cols_r)
    calc_std_dev_r = calc_std_dev(sum_cols_r)
    calc_variance_r = calc_variance(sum_cols_r)
    calc_median_r = calc_median(sum_cols_r)

    load_results_r = load_results(pick_least_r, pick_greatest_r, calc_mean_r, calc_std_dev_r, calc_variance_r,
                                  calc_median_r)

    kickoff_dag = DummyOperator(task_id="kickoff_dag")
    complete_dag = DummyOperator(task_id="complete_dag")

    kickoff_dag >> build_raw_df
    load_results_r >> complete_dag


taskflow_v6d_dag = taskflow_v6d()
