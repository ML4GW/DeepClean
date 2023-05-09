from exporter.exporter import Exporter
from flask import Flask

exporter = Exporter.from_config()
app = Flask(__name__)


@app.route("/export/<train_dir>")
def export_model(train_dir):
    app.logger.info(f"Received export request for {train_dir}")
    exporter.export_weights(train_dir)
    return "", 200


@app.route("/increment")
def increment_ensemble():
    app.logger.info("Received ensemble version increment request")
    exporter.update_ensemble_versions()
    return "", 200


@app.route("/production-version")
def get_production_version():
    version = exporter.get_production_version()
    app.logger.info(f"Reporting production DeepClean version {version}")
    return str(version), 200


@app.route("/latest-version")
def get_latest_version():
    version = max(exporter.repo.models["deepclean"].versions)
    app.logger.info(f"Reporting latest DeepClean version {version}")
    return str(version), 200
