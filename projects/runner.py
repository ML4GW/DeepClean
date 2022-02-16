import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml
from clikit.io.console_io import ConsoleIO
from conda.cli import python_api as conda
from poetry.factory import Factory
from poetry.utils.env import EnvManager


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
class PoetryEnvironment:
    project: "Project"

    def __post_init__(self):
        self._poetry = Factory().create_poetry(self.project.path)
        self._manager = EnvManager(self._poetry)
        self._venv = self._manager.create_venv(ConsoleIO())

    def exists(self):
        return True

    def create(self):
        return

    def contains(self, project):
        name = project.name.replace("-", "_")
        return self._venv.site_packages.find_distribution(name) is None

    def run(self, *args):
        try:
            return self._venv.run(*args)
        except Exception as e:
            raise RuntimeError(
                "Executing command {} in poetry environment {} "
                "failed with error:\n{}".format(
                    args, self._venv.path.name, str(e)
                )
            )


@dataclass
class CondaEnvironment:
    name: str

    def run_command(self, *args):
        stdout, stderr, exit_code = conda.run_command(*args)
        if exit_code:
            raise RuntimeError(
                "Executing command {} in conda environment {} "
                "failed with error:\n{}".format(args, self.name, stderr)
            )
        return stdout

    def exists(self):
        stdout = self.run_command(conda.Commands.INFO, "--envs")
        env_names = [
            row.split()[0]
            for row in stdout.splitlines()
            if row and not row.startswith("#")
        ]
        return self.name in env_names

    def create(self):
        base_env = CondaEnvironment("deepclean-base")
        if not base_env.exists():
            logging.info("Creating base DeepClean environment")
            stdout = self.run_command(
                conda.Commands.CREATE,
                "-f",
                str(Path.cwd().parent / "environment.yaml"),
            )
            logging.info(stdout)

        stdout = self.run_command(
            conda.Commands.CREATE, "-n", self.name, "--clone", "deepclean-base"
        )
        logging.info(stdout)

    def run(self, *args):
        return self.run_command(conda.Commands.RUN, "-n", self.name, *args)


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
                return not poetry_config["virtualenvs"]["create"]
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

    def create_venv(self):
        if self.uses_conda():
            env_name = self.runner_config.get(
                "conda_env", f"deepclean-{self.name}"
            )
            venv = CondaEnvironment(env_name)
            if not venv.exists():
                venv.create()
        else:
            venv = PoetryEnvironment(self)

        # TODO: ensure environment has this project
        # installed somewhere
        # if not venv.contains(self):
        #    venv.install()
        return venv

    def execute(self, command: str, subcommand: Optional[str] = None):
        venv = self.create_venv()
        typeo_arg = str(self.pipeline.path)
        try:
            if command in self.pipeline.typeo_config["scripts"]:
                typeo_arg += ":" + command
        except KeyError:
            if subcommand is not None:
                typeo_arg += "::" + subcommand
        else:
            if subcommand is not None:
                typeo_arg += ":" + subcommand

        venv.run(command, "--typeo", typeo_arg)


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
        stdout = project.execute(command, subcommand)
        logging.info(stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=str, help="Pipeline to run")
    parser.add_argument("--log-file", type=str, help="Path to write logs to")
    parser.add_argument("--verbose", action="store_true", help="Log verbosity")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if args.log_file is not None:
        handler = logging.FileHandler(filename=args.log_file, mode="w")

    main(args.pipeline)
