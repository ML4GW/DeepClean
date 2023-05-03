from exporter.exporter import Exporter
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2 as model_config


def load_config(model):
    config = model_config.ModelConfig()
    with open(model / "config.pbtxt", "r") as f:
        text_format.Merge(f.read(), config)
    return config


class TestExporter:
    def test_init(self, tmp_path):
        tmp_path.mkdir(exist_ok=True, parents=True)
        _ = Exporter(
            tmp_path,
            channels=list("DCAEB"),
            kernel_length=1,
            sample_rate=4096,
            inference_sampling_rate=512,
            batch_size=128,
            aggregation_time=0.25,
            streams_per_gpu=2,
            instances=4,
        )

        deepclean_path = tmp_path / "deepclean"
        for fname in deepclean_path.iterdir():
            assert fname.name in ("1", "config.pbtxt")
        deepclean_config = load_config(deepclean_path)

        assert deepclean_config.input[0].dims == [128, 4, 4096]
        assert deepclean_config.output[0].dims == [128, 4096]
        assert deepclean_config.instance_group[0].count == 4
        assert deepclean_config.parameters["channels"].string_value == (
            "A,B,C,E"
        )
