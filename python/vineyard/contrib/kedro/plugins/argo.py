#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is referred and modified from Kedro project:
#
#    https://github.com/kedro-org/kedro/blob/main/docs/source/deployment/argo.md
#
# which is licensed under the Apache License, Version 2.0.

"""Command line `kedro vineyard argo generate` will generate an Argo workflow
   configuration file from a Kedro pipeline.
"""

import re
from pathlib import Path

import click
from jinja2 import Environment
from jinja2 import FileSystemLoader
from kedro.framework.cli.utils import command_with_verbosity
from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

from vineyard.contrib.kedro.plugins.cli import vineyard as vineyard_cli

TEMPLATE_FILE = "argo_spec.tmpl"
TEMPLATE_PATH = Path("templates")


@vineyard_cli.group()
def argo():
    """Commands for working with argo."""


@command_with_verbosity(argo, "generate")
@click.option("--image", "-i", "image", type=str, required=True, default="")
@click.option("--pipeline", "-p", "pipeline_name", type=str, default=None)
@click.option(
    "--vineyard", "-v", "vineyardd_name", type=str, default="vineyardd-sample"
)
@click.option(
    "--namespace", "-n", "vineyardd_namespace", type=str, default="vineyard-system"
)
@click.option(
    "--with_vineyard_operator",
    "-w",
    "with_vineyard_operator",
    type=bool,
    default=True,
    help="Whether to generate the workflow with vineyard operator.",
)
@click.option("--output_path", "-o", "output_path", type=str, default=".")
def generate_argo_config(
    image,
    pipeline_name,
    vineyardd_name,
    vineyardd_namespace,
    with_vineyard_operator,
    output_path,
    verbose,
):
    """Generates an Argo workflow configuration file from a Kedro pipeline.

    Args:
        image (str, required): The Docker image to use for the workflow.

        pipeline_name (str, optional): The name of the pipeline to
        generate the workflow for. If not specified, the default pipeline
        will be used.

        vineyardd_name (str, optional): The name of the Vineyardd server
        to use. Defaults to "vineyardd-sample".

        vineyardd_namespace (str, optional): The namespace of the Vineyardd
        server. Defaults to "vineyard-system".

        with_vineyard_operator (bool, optional): Whether to generate the
        workflow with vineyard operator. Defaults to True.

        output_path (str, optional): The path to the output file. Defaults
        to the current directory.

    """
    # get the absolute path of the template directory
    template_path = Path(__file__).resolve().parent / TEMPLATE_PATH

    loader = FileSystemLoader(searchpath=template_path)
    template_env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = template_env.get_template(TEMPLATE_FILE)

    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    package_name = metadata.package_name

    pipeline_name = pipeline_name or "__default__"
    pipeline = pipelines.get(pipeline_name)

    tasks = get_dependencies(pipeline.node_dependencies)

    output = template.render(
        image=image,
        package_name=package_name,
        tasks=tasks,
        vineyardd_name=vineyardd_name,
        vineyardd_namespace=vineyardd_namespace,
        with_vineyard_operator=with_vineyard_operator,
    )

    output_path = Path(output_path)

    (output_path / f"argo-{package_name}.yml").write_text(output)


def get_dependencies(dependencies):
    """Gets the dependencies of a Kedro pipeline.

    Args:
        dependencies (dict): A dictionary of nodes to their parent nodes.

    Returns:
        list: A list of dictionaries representing the dependencies of the pipeline.
    """
    deps_dict = [
        {
            "node": node.name,
            "name": clean_name(node.name),
            "deps": [clean_name(val.name) for val in parent_nodes],
        }
        for node, parent_nodes in dependencies.items()
    ]
    return deps_dict


def clean_name(name):
    return re.sub(r"[\W_]+", "-", name).strip("-")
