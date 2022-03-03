import argparse
import logging
import sys

from runner.project import Pipeline, Project


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

        project = pipeline.create_project(component)
        stdout = pipeline.run_step(project, command, subcommand)
        logging.info(stdout)


def build(project_path: str):
    project = Project(project_path)
    project.create_venv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, help="Path to write logs to")
    parser.add_argument("--verbose", action="store_true", help="Log verbosity")

    subparsers = parser.add_subparsers(dest="subcommand")
    subparser = subparsers.add_parser("run", description="Run a pipeline")
    subparser.add_argument("pipeline", type=str, help="Pipeline to run")

    subparser = subparsers.add_parser("build", description="Build a project")
    subparser.add_argument("project-path", type=str, help="Project to build")

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
    if args.subcommand == "build":
        build(args.project_path)


if __name__ == "__main__":
    main()
