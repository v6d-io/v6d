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
import shutil
import sys
from importlib import import_module
from pathlib import Path
from sys import version_info
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import click
from click import secho
from kedro import __version__ as kedro_version
from kedro.framework.cli.utils import KedroCliError
from kedro.framework.cli.utils import call
from semver import VersionInfo

from .cli import vineyard as vineyard_cli

KEDRO_VERSION = VersionInfo.parse(kedro_version)
TEMPLATE_PATH = Path("templates")
DOCKER_FILE = "Dockerfile"
DEFAULT_BASE_IMAGE = f"python:{version_info.major}.{version_info.minor}-slim"


@vineyard_cli.group()
def docker():
    """Commands for working with docker."""


@docker.command("init")
def docker_init():
    """Initialize a Dockerfile for the project."""
    project_path = Path.cwd()

    if KEDRO_VERSION.match(">=0.17.0"):
        verbose = KedroCliError.VERBOSE_ERROR
    else:
        from kedro.framework.cli.cli import (
            _VERBOSE as verbose,
        )  # noqa # pylint:disable=import-outside-toplevel, no-name-in-module

    # get the absolute path of the template directory
    template_path = Path(__file__).resolve().parent / TEMPLATE_PATH

    copy_template_files(
        project_path,
        template_path,
        ["Dockerfile"],
        verbose,
    )


@docker.command("build")
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
@click.pass_context
def docker_build(
    ctx, uid, gid, base_image, image, docker_args
):  # pylint: disable=too-many-arguments
    """Build a Docker image for the project."""
    uid, gid = get_uid_gid(uid, gid)
    project_path = Path.cwd()
    image = image or project_path.name

    ctx.invoke(docker_init)

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


def copy_template_files(
    project_path: Path,
    template_path: Path,
    template_files: Sequence[str],
    verbose: bool = False,
):
    """
    If necessary copy files from a template directory into a project directory.

    Args:
        project_path: Destination path.
        template_path: Source path.
        template_files: Files to copy.
        verbose: Echo the names of any created files.

    """
    for file_ in template_files:
        dest_file = "Dockerfile" if file_.startswith("Dockerfile") else file_
        dest = project_path / dest_file
        if not dest.exists():
            src = template_path / file_
            shutil.copyfile(str(src), str(dest))
            if verbose:
                secho(f"Creating `{dest}`")
        else:
            msg = f"{dest_file} already exists and won't be overwritten."
            secho(msg, fg="yellow")


def get_uid_gid(uid: int = None, gid: int = None) -> Tuple[int, int]:
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
