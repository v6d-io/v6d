"""
This is a boilerplate pipeline
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=remove_outliers,
                inputs=["house_prices", "parameters"],
                outputs="house_prices_no_outliers",
                name="outliers_node",
            ),
            node(
                func=create_target,
                inputs="house_prices_no_outliers",
                outputs="y_train",
                name="create_target_node",
            ),
            node(
                func=drop_cols,
                inputs=["house_prices_no_outliers", "parameters"],
                outputs="house_prices_drop",
                name="drop_cols_node",
            ),
            node(
                func=fill_na,
                inputs=["house_prices_drop", "parameters"],
                outputs="house_prices_no_na",
                name="fill_na_node",
            ),
            node(
                func=total_sf,
                inputs="house_prices_drop",
                outputs="house_prices_clean",
                name="total_sf_node",
            ),
        ]
    )