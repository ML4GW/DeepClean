from exporter.exporter import Exporter
from flask import Flask

exporter = Exporter.from_config()
app = Flask(__name__)


@app.route("/export/<train_dir>")
def export_model(train_dir):
    app.logger.info(f"Received export request for {train_dir}")
    exporter.export_weights(train_dir)


@app.route("/increment")
def increment_ensemble():
    app.logger.info("Received ensemble version increment request")
    exporter.update_ensemble_version()
