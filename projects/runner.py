import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml
from conda.cli import python_api as conda


@dataclass
class Pipeline:
    name: str

    @property
    def path(self):
        return Path(".").resolve() / self.name

    @property
    def config(self):
        return self._config.copy()

    @property
    def steps(self):
        return self._config["tool"]["runner"]["steps"]

    @property
    def typeo_config(self):
        return self._config["tool"]["typeo"]

    def __post_init__(self):
        config_path = self.path / "pyproject.toml"
        try:
            with open(self.path / "pyproject.toml", "r") as f:
                self._config = toml.load(f)
        except FileNotFoundError:
            raise ValueError(
                "Pipeline {} has no associated 'pyproject.toml' "
                "at location {}".format(self.name, config_path)
            )

        try:
            _ = self.steps
        except KeyError:
            raise ValueError(
                f"Config file {config_path} has no '[tool.runner]' "
                "table or 'steps' key in it."
            )
        try:
            _ = self.typeo_config
        except KeyError:
            raise ValueError(
                f"Config file {config_path} has no '[tool.typeo]' "
                "table necessary to run projects."
            )


@dataclass
class Project:
    name: str
    pipeline: Pipeline

    @property
    def path(self):
        return self.pipeline.path / self.name

    @property
    def config(self):
        return self._config.copy()

    @property
    def runner_config(self):
        try:
            return self.config["tool"]["runner"]
        except KeyError:
            return {}

    def uses_conda(self):
        try:
            with open(self.path / "poetry.toml", "r") as f:
                poetry_config = toml.load(f)
        except FileNotFoundError:
            return False
        else:
            try:
                return not poetry_config["virtualens"]["create"]
            except KeyError:
                return False

    def __post_init__(self):
        config_path = self.path / "pyproject.toml"
        try:
            with open(self.path / "pyproject.toml", "r") as f:
                self._config = toml.load(f)
        except FileNotFoundError:
            raise ValueError(
                "Project {} has no associated 'pyproject.toml' "
                "at location {}".format(self.name, config_path)
            )

    def execute(self, command: str, subcommand: Optional[str] = None):
        typeo_arg = self.pipeline.path
        try:
            if command in self.pipeline.typeo_config["scripts"]:
                typeo_arg += ":" + command
        except KeyError:
            if subcommand is not None:
                typeo_arg += "::" + subcommand
        else:
            if subcommand is not None:
                typeo_arg += ":" + subcommand

        if self.uses_conda():
            env_name = self.runner_config.get(
                "conda_env", f"deepclean-{self.name}"
            )

            stdout, stderr, exit_code = conda.run_command(
                conda.Commands.RUN,
                "-n",
                env_name,
                command,
                "--typeo",
                typeo_arg,
            )
            if not exit_code:
                raise RuntimeError(
                    f"Conda command execution failed with error:\n{stderr}"
                )
            return stdout


def main(pipeline: str):
    pipeline = Pipeline(pipeline)

    for step in pipeline.steps:
        try:
            component, command, subcommand = step.split(":")
        except ValueError:
            try:
                component, command = step.split(":")
                subcommand = None
            except ValueError:
                raise ValueError(f"Can't parse pipeline step '{step}'")

        project = Project(component)
        project.execute(command, subcommand)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=str, help="Pipeline to run")
    args = parser.parse_args()
    main(args.pipeline)
