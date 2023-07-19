#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is referred and modified from Kedro project:
#
#    https://github.com/kedro-org/kedro/blob/main/kedro/framework/cli/catalog.py
#
# which is licensed under the Apache License, Version 2.0.

"""Command line `kedro vineyard catalog {list/create}` will list the catalog of
   given pipelines, and create a catalog configuration which overrides existing
   ones with VineyardDataSet.
"""

import os
import warnings

import click
import yaml
from click import secho
from kedro.framework.cli.catalog import _create_session
from kedro.framework.cli.catalog import _map_type_to_datasets
from kedro.framework.cli.utils import KedroCliError
from kedro.framework.cli.utils import command_with_verbosity
from kedro.framework.cli.utils import env_option
from kedro.framework.cli.utils import split_string
from kedro.framework.project import pipelines
from kedro.framework.project import settings
from kedro.framework.startup import ProjectMetadata

from vineyard.contrib.kedro.plugins.cli import vineyard as vineyard_cli


@vineyard_cli.group()
def catalog():
    """Commands for working with catalog."""


@command_with_verbosity(catalog, 'list')
@env_option
@click.option(
    "--pipeline",
    "-p",
    type=str,
    default="",
    help="Name of the modular pipeline to run. If not set, "
    "the project pipeline is run by default.",
    callback=split_string,
)
@click.pass_obj
def list_datasets(metadata: ProjectMetadata, pipeline, env, verbose):
    """Show datasets per type."""
    title = "DataSets in '{}' pipeline"
    not_mentioned = "Datasets not mentioned in pipeline"
    mentioned = "Datasets mentioned in pipeline"

    session = _create_session(metadata.package_name, env=env)
    context = session.load_context()
    datasets_meta = context.catalog._data_sets  # pylint: disable=protected-access
    catalog_ds = set(context.catalog.list())

    target_pipelines = pipeline or pipelines.keys()

    result = {}
    for pipe in target_pipelines:
        pl_obj = pipelines.get(pipe)
        if pl_obj:
            pipeline_ds = pl_obj.data_sets()
        else:
            existing_pls = ", ".join(sorted(pipelines.keys()))
            raise KedroCliError(
                f"'{pipe}' pipeline not found! Existing pipelines: {existing_pls}"
            )

        unused_ds = catalog_ds - pipeline_ds
        default_ds = pipeline_ds - catalog_ds
        used_ds = catalog_ds - unused_ds

        unused_by_type = _map_type_to_datasets(unused_ds, datasets_meta)
        used_by_type = _map_type_to_datasets(used_ds, datasets_meta)

        if default_ds:
            used_by_type["DefaultDataSet"].extend(default_ds)

        data = ((not_mentioned, dict(unused_by_type)), (mentioned, dict(used_by_type)))
        result[title.format(pipe)] = {key: value for key, value in data if value}

    secho(yaml.dump(result))


@command_with_verbosity(catalog, 'create')
@env_option(
    default='vineyard',
    help="Environment to create Data Catalog YAML file in. Defaults to `vineyard`.",
)
@click.option(
    "--pipeline",
    "-p",
    "pipeline_name",
    type=str,
    required=True,
    help="Name of a pipeline.",
)
@click.pass_obj
def create_catalog(metadata: ProjectMetadata, pipeline_name, env, verbose):
    """Create Data Catalog YAML configuration with missing datasets.

    Add `MemoryDataSet` datasets to Data Catalog YAML configuration file
    for each dataset in a registered pipeline if it is missing from
    the `DataCatalog`.

    The catalog configuration will be saved to
    `<conf_source>/<env>/catalog/<pipeline_name>.yml` file.
    """
    env = env or "base"

    # ensure the config directory for given env exists
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        session = _create_session(metadata.package_name, env='base')
    context = session.load_context()
    os.makedirs(context.project_path / settings.CONF_SOURCE / env, exist_ok=True)

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    session = _create_session(metadata.package_name, env=env)
    context = session.load_context()

    pipeline = pipelines.get(pipeline_name)

    if not pipeline:
        existing_pipelines = ", ".join(sorted(pipelines.keys()))  # noqa: F841
        raise KedroCliError(
            f"'{pipeline_name}' pipeline not found! "
            "Existing pipelines: {existing_pipelines}"
        )

    pipe_datasets = {
        ds_name
        for ds_name in pipeline.data_sets()
        if not ds_name.startswith("params:") and ds_name != "parameters"
    }

    catalog_datasets = {
        ds_name
        for ds_name in context.catalog._data_sets.keys()
        if not ds_name.startswith("params:") and ds_name != "parameters"
    }

    # Datasets that are missing in Data Catalog
    missing_ds = sorted(pipe_datasets - catalog_datasets)
    if missing_ds:
        catalog_path = (
            context.project_path
            / settings.CONF_SOURCE
            / env
            / "catalog"
            / f"{pipeline_name}.yml"
        )
        _add_missing_datasets_to_catalog(missing_ds, catalog_path)
        click.echo(f"Data Catalog YAML configuration was created: {catalog_path}")
    else:
        click.echo("All datasets are already configured.")


def _add_missing_datasets_to_catalog(missing_ds, catalog_path):
    from vineyard.contrib.kedro.io import VineyardDataSet

    if catalog_path.is_file():
        catalog_config = yaml.safe_load(catalog_path.read_text()) or {}
    else:
        catalog_config = {}

    ds_type = VineyardDataSet.__module__ + '.' + VineyardDataSet.__qualname__
    for ds_name in missing_ds:
        catalog_config[ds_name] = {"type": ds_type, "ds_name": ds_name}

    # Create only `catalog` folder under existing environment
    # (all parent folders must exist).
    catalog_path.parent.mkdir(exist_ok=True)
    with catalog_path.open(mode="w") as catalog_file:
        yaml.safe_dump(catalog_config, catalog_file, default_flow_style=False)
