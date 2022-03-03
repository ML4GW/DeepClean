import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml
from runner.env import CondaEnvironment, PoetryEnvironment


@dataclass
class ProjectBase:
    path: str

    def __post_init__(self):
        self.path = Path(os.path.abspath(self.path))
        config_path = self.path / "pyproject.toml"
        try:
            with open(self.path / "pyproject.toml", "r") as f:
                self._config = toml.load(f)
        except FileNotFoundError:
            raise ValueError(
                "{} {} has no associated 'pyproject.toml' "
                "at location {}".format(
                    self.__class__.__name__, self.path, config_path
                )
            )

    @property
    def config(self):
        return self._config.copy()


@dataclass
class Project(ProjectBase):
    def __post_init__(self):
        super().__post_init__()
        self.name = self._config["tool"]["poetry"]["name"]
        self._venv = None

    @property
    def runner_config(self):
        try:
            return self.config["tool"]["runner"]
        except KeyError:
            return {}

    @property
    def venv(self):
        return self._venv

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

    def create_venv(self, force: bool = False):
        if self.uses_conda():
            env_name = self.runner_config.get(
                "conda_env", f"deepclean-{self.name}"
            )
            venv = CondaEnvironment(env_name)
            if not venv.exists():
                venv.create()
        else:
            venv = PoetryEnvironment(self.path)
            venv.create()

        # ensure environment has this project
        # installed somewhere
        if not venv.contains(self.name):
            logging.info(
                "Installing project '{}' from '{}' into "
                "virtual environemnt '{}'".format(
                    self.name, self.path, venv.name
                )
            )
            venv.install(self.path)
        elif force:
            logging.info(
                "Updating project '{}' from '{}' in "
                "virtual environment '{}'".format(
                    self.name, self.path, venv.name
                )
            )
            venv.install(self.path)
        else:
            logging.info(
                "Project '{}' at '{}' already installed in "
                "virtual environment '{}'".format(
                    self.name, self.path, venv.name
                )
            )

        self._venv = venv

    def run(self, *args):
        if self.venv is None:
            self.create_venv()
        self.venv.run(*args)


@dataclass
class Pipeline:
    def __post_init__(self):
        super().__post_init__()

        config_path = self.path / "pyproject.toml"
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

    @property
    def steps(self):
        return self.config["tool"]["runner"]["steps"]

    @property
    def typeo_config(self):
        return self.config["tool"]["typeo"]

    def create_project(self, name):
        return Project(self.path / name)

    def run_step(
        self, project: Project, command: str, subcommand: Optional[str] = None
    ):
        typeo_arg = str(self.path)
        try:
            if command in self.typeo_config["scripts"]:
                typeo_arg += ":" + command
        except KeyError:
            if subcommand is not None:
                typeo_arg += "::" + subcommand
        else:
            if subcommand is not None:
                typeo_arg += ":" + subcommand

        project.run(command, "--typeo", typeo_arg)
