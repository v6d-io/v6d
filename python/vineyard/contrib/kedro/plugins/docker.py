#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is referred and modified from Kedro project:
#
#    https://github.com/kedro-org/kedro-plugins/blob/main/kedro-docker/kedro_docker/plugin.py
#
# which is licensed under the Apache License, Version 2.0.

"""Command line `kedro vineyard docker` will containize the kedro project"""

import os
from importlib import import_module
from pathlib import Path
from sys import version_info
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import click
from jinja2 import Environment
from jinja2 import FileSystemLoader
from kedro.framework.cli.utils import call
from kedro.framework.cli.utils import command_with_verbosity

from vineyard.contrib.kedro.plugins.cli import vineyard as vineyard_cli

TEMPLATE_PATH = Path("templates")
DOCKER_FILE_TMPL = "Dockerfile.tmpl"
DEFAULT_BASE_IMAGE = f"python:{version_info.major}.{version_info.minor}-slim"


@vineyard_cli.group()
def docker():
    """Commands for working with docker."""


@command_with_verbosity(docker, 'init')
@click.option(
    "--with-vineyard",
    "",
    "with_vineyard",
    type=bool,
    default=True,
    help="Whether to install vineyard dependencies in the dockerfile.",
)
def docker_init(with_vineyard, verbose):
    """Initialize a Dockerfile for the project."""
    project_path = Path.cwd()

    # get the absolute path of the template directory
    template_path = Path(__file__).resolve().parent / TEMPLATE_PATH

    generate_dockerfile(
        project_path,
        template_path,
        with_vineyard=with_vineyard,
    )


@command_with_verbosity(docker, 'build')
@click.option(
    "--uid",
    type=int,
    default=None,
    help="User ID for kedro user inside the container. "
    "Default is the current user's UID",
)
@click.option(
    "--gid",
    type=int,
    default=None,
    help="Group ID for kedro user inside the container. "
    "Default is the current user's GID",
)
@click.option(
    "--base-image",
    type=str,
    default=DEFAULT_BASE_IMAGE,
    show_default=True,
    help="Base image for Dockerfile.",
)
@click.option(
    "--image",
    type=str,
    default=None,
    show_default=True,
    help="Name of the image to build.",
)
@click.option(
    "--docker-args",
    type=str,
    default=None,
    show_default=True,
    help="Extra arguments to pass to `docker build` command.",
)
@click.option(
    "--with-vineyard",
    "",
    "with_vineyard",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to install vineyard dependencies in the dockerfile.",
)
@click.pass_context
def docker_build(
    ctx,
    uid,
    gid,
    base_image,
    image,
    docker_args,
    with_vineyard,
    verbose,
):  # pylint: disable=too-many-arguments
    """Build a Docker image for the project."""
    uid, gid = get_uid_gid(uid, gid)
    project_path = Path.cwd()
    image = image or project_path.name

    ctx.invoke(docker_init, with_vineyard=with_vineyard)

    combined_args = compose_docker_run_args(
        required_args=[
            ("--build-arg", f"KEDRO_UID={uid}"),
            ("--build-arg", f"KEDRO_GID={gid}"),
            ("--build-arg", f"BASE_IMAGE={base_image}"),
        ],
        # add image tag if only it is not already supplied by the user
        optional_args=[("-t", image)],
        user_args=docker_args,
    )
    command = ["docker", "build"] + combined_args + [str(project_path)]
    call(command)


def generate_dockerfile(
    project_path: Path,
    template_path: Path,
    with_vineyard: bool = False,
):
    """generate the Dockerfile for the project.

    Args:
        project_path (Path): Destination path.
        template_path: Source path.
        with_vineyard (bool, optional): The dockerfile with vineyard or not.
                                        Defaults to False.
    """
    # get the absolute path of the template directory
    template_path = Path(__file__).resolve().parent / TEMPLATE_PATH

    loader = FileSystemLoader(searchpath=template_path)
    template_env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = template_env.get_template(DOCKER_FILE_TMPL)

    dockerfile = template.render(
        with_vineyard=with_vineyard,
    )

    dest = project_path / "Dockerfile"

    if dest.exists():
        print(f"{dest} already exists and won't be overwritten.")
    else:
        # Create the Dockerfile in the destination path
        with open(dest, "w", encoding="utf-8") as f:
            f.write(dockerfile)

        print(f"Created `{dest}`")


def get_uid_gid(
    uid: Optional[int] = None, gid: Optional[int] = None
) -> Tuple[int, int]:
    """
    Get UID and GID to be passed into the Docker container.
    Defaults to the current user's UID and GID on Unix and (999, 0) on Windows.

    Args:
        uid: Input UID.
        gid: Input GID.

    Returns:
        (UID, GID).
    """

    # Default uid 999 is chosen as the one having potentially the lowest chance
    # of clashing with some existing user in the Docker container.
    _default_uid = 999

    # Default gid 0 corresponds to the root group.
    _default_gid = 0

    if uid is None:
        uid = os.getuid() if os.name == "posix" else _default_uid

    if gid is None:
        gid = (
            import_module("pwd").getpwuid(uid).pw_gid
            if os.name == "posix"
            else _default_gid
        )

    return uid, gid


# pylint: disable=too-many-arguments
def compose_docker_run_args(
    required_args: Sequence[Tuple[str, Union[str, None]]] = None,
    optional_args: Sequence[Tuple[str, Union[str, None]]] = None,
    user_args: Sequence[str] = None,
) -> List[str]:
    """
    Make a list of arguments for the docker command.

    Args:
        required_args: List of required arguments.
        optional_args: List of optional arguments, these will be added if only
            not present in `user_args` list.
        user_args: List of arguments already specified by the user.

    Returns:
        List of arguments for the docker command.
    """

    required_args = required_args or []
    optional_args = optional_args or []
    user_args = user_args or []
    split_user_args = {ua.split("=", 1)[0] for ua in user_args}

    def _add_args(name_: str, value_: str = None, force_: bool = False) -> List[str]:
        """
        Add extra args to existing list of CLI args.
        Args:
            name_: Arg name to add.
            value_: Arg value to add, skipped if None.
            force_: Add the argument even if it's present in the current list of args.

        Returns:
            List containing the new args and (optionally) its value or an empty list
                if no values to be added.
        """
        if not force_ and name_ in split_user_args:
            return []
        return [name_] if value_ is None else [name_, value_]

    combined_args = []
    for arg_name, arg_value in required_args:
        combined_args += _add_args(arg_name, arg_value, True)
    for arg_name, arg_value in optional_args:
        combined_args += _add_args(arg_name, arg_value)
    return combined_args + user_args
