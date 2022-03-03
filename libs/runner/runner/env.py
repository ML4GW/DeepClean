import logging
import re
from dataclasses import dataclass
from pathlib import Path

from cleo.application import Application
from conda.cli import python_api as conda
from poetry.factory import Factory
from poetry.installation.installer import Installer
from poetry.masonry.builders import EditableBuilder
from poetry.utils.env import EnvManager


@dataclass
class PoetryEnvironment:
    path: str

    def __post_init__(self):
        self._poetry = Factory().create_poetry(self.path)
        self._manager = EnvManager(self._poetry)
        self._io = Application.create_io(self)
        self._venv = None

    @property
    def name(self) -> str:
        return self._venv.path.name

    def exists(self) -> bool:
        return self.manager.get() != self.manager.get_system_env()

    def create(self):
        self._venv = self._manager.create_venv(self._io)
        return self._venv

    def contains(self, project: str) -> bool:
        name = project.replace("-", "_")
        return self._venv.site_packages.find_distribution(name) is not None

    def install(self, *args) -> None:
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

    def contains(self, project_name: str) -> bool:
        regex = re.compile(f"(?m)^{project_name} ")
        package_list = self.run_command(conda.Commands.LIST, "-n", self.name)
        return regex.search(package_list) is not None

    def install(self, project_path: str):
        self.run("/bin/bash", "-c", f"cd {project_path} && poetry install")

    def run(self, *args):
        return self.run_command(conda.Commands.RUN, "-n", self.name, *args)
