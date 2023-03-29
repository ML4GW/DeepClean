from flask import Flask
from export.exporter import Exporter


app = Flask(__name__)
exporter = Exporter.from_config()

@app.route("/")
def export_model(train_dir):
    exporter.export_version(train_dir)
