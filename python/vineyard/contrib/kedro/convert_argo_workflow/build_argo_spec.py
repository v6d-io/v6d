import re
from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

TEMPLATE_FILE = "argo_spec.tmpl"
SEARCH_PATH = Path("templates")

@click.command()
@click.argument("image", required=True)
@click.option("-p", "--pipeline", "pipeline_name", default=None)
@click.option("--env", "-e", type=str, default=None)
def generate_argo_config(image, pipeline_name, env):
    loader = FileSystemLoader(searchpath=SEARCH_PATH)
    template_env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = template_env.get_template(TEMPLATE_FILE)

    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    package_name = metadata.package_name

    pipeline_name = pipeline_name or "__default__"
    pipeline = pipelines.get(pipeline_name)

    tasks = get_dependencies(pipeline.node_dependencies)

    output = template.render(image=image, package_name=package_name, tasks=tasks)

    (SEARCH_PATH / f"argo-{package_name}.yml").write_text(output)


def get_dependencies(dependencies):
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


if __name__ == "__main__":
    generate_argo_config()
