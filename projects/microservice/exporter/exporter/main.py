from flask import Flask
from exporter.exporter import Exporter

app = Flask(__name__)
exporter = Exporter.from_config()


@app.route("/export/<train_dir>")
def export_model(train_dir):
    exporter.export_weights(train_dir)


@app.route("/increment")
def increment_ensemble():
    exporter.update_ensemble_version()
