import base64
import io
import json
from unittest import mock

from integrations.uavflow_sim.policy_proxy import Proxy


def test_predict_maps_actions():
    png = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    payload = {"image": base64.b64encode(png.getvalue()).decode(), "proprio": [0, 0, -10, 0, 0, 0], "instr": "approach"}
    response = mock.Mock()
    response.__enter__ = mock.Mock(return_value=response)
    response.__exit__ = mock.Mock(return_value=False)
    response.read.return_value = json.dumps({"actions": [[1, 0, 0, 0]]}).encode()
    with mock.patch("urllib.request.urlopen", return_value=response):
        out = Proxy("http://example").predict(payload)
    assert out["action"] == [[1, 0, 0, 0]]
