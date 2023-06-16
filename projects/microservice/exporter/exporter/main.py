from exporter.exporter import Exporter
from flask import Flask

exporter = Exporter.from_config()
app = Flask(__name__)


@app.route("/alive")
def alive():
    return "0", 200


@app.route("/export/<train_dir>")
def export_model(train_dir):
    app.logger.info(f"Received export request for {train_dir}")
    exporter.export_weights(train_dir)
    return "", 200


@app.route("/production-version/set/<version>")
def set_production_version(version):
    version = int(version)
    app.logger.info(f"Setting production DeepClean to version {version}")
    if version == -1:
        version = None
    version = exporter.update_ensemble_versions(version)
    return str(version), 200


@app.route("/production-version")
def get_production_version():
    version = exporter.get_production_version()
    app.logger.info(f"Reporting production DeepClean version {version}")
    return str(version), 200


@app.route("/latest-version")
def get_latest_version():
    version = exporter.latest_version
    app.logger.info(f"Reporting latest DeepClean version {version}")
    return str(version), 200
