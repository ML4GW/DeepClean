import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml
from cleo.application import Application
from conda.cli import python_api as conda
from poetry.factory import Factory
from poetry.installation.installer import Installer
from poetry.masonry.builders import EditableBuilder
from poetry.utils.env import EnvManager


@dataclass
class Pipeline:
    path: str
 
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
        self.path = Path(os.path.abspath(self.path))
        config_path = self.path / "pyproject.toml"
        try:
            with open(self.path / "pyproject.toml", "r") as f:
                self._config = toml.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"Pipeline {self.path} has no associated 'pyproject.toml'"
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
        self._io = Application.create_io(self)
        self._venv = self._manager.create_venv(self._io)

    @property
    def name(self):
        return self._venv.path.name

    def exists(self):
        return True

    def create(self):
        return

    def contains(self, project):
        name = project.name.replace("-", "_")
        return self._venv.site_packages.find_distribution(name) is not None

    def install(self):
        installer = Installer(
            self._io,
            self._venv,
            self._poetry.package,
            self._poetry.locker,
            self._poetry.pool,
            self._poetry.config,
        )
        installer.update(True)
        installer.use_executor(True)
        installer.run()

        builder = EditableBuilder(self._poetry, self._venv, self._io)
        builder.build()

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

    def contains(self, project):
        # TODO: implement
        return True

    def install(self, project):
        # TODO: implement
        raise NotImplementedError

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

        # ensure environment has this project
        # installed somewhere
        # TODO: this won't work for conda environments
        # at the moment
        if not venv.contains(self):
            logging.info(
                f"Installing project '{self.name}' into "
                f"virtual environment '{venv.name}'"
            )
            venv.install()
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


def run(pipeline: str):
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

        project = Project(component, pipeline)
        stdout = project.execute(command, subcommand)
        logging.info(stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, help="Path to write logs to")
    parser.add_argument("--verbose", action="store_true", help="Log verbosity")

    subparsers = parser.add_subparsers(dest="subcommand")
    subparser = subparsers.add_parser("run")
    subparser.add_argument("pipeline", type=str, help="Pipeline to run")

    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if args.log_file is not None:
        handler = logging.FileHandler(filename=args.log_file, mode="w")
        logging.getLogger().addHandler(handler)

    if args.subcommand == "run":
        run(args.pipeline)


if __name__ == "__main__":
    main()
