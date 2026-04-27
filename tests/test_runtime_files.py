import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeFilesTest(unittest.TestCase):
    def test_runtime_config_threshold_order(self) -> None:
        config_path = REPO_ROOT / "config" / "runtime_config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        detection = data["detection"]
        self.assertLessEqual(detection["detector_conf_threshold"], detection["tracker_low_conf_threshold"])
        self.assertLessEqual(detection["tracker_low_conf_threshold"], detection["tracker_high_conf_threshold"])
        self.assertLessEqual(detection["tracker_high_conf_threshold"], 1.0)
        self.assertLessEqual(detection["nms_threshold"], 1.0)

    def test_runtime_paths_exist(self) -> None:
        config_path = REPO_ROOT / "config" / "runtime_config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        runtime = data["runtime"]

        self.assertTrue((REPO_ROOT / runtime["worker_script"]).exists())
        self.assertTrue((REPO_ROOT / runtime["package_dir"]).exists())

    def test_manifest_has_expected_fields(self) -> None:
        manifest_path = REPO_ROOT / "deliverables" / "model-manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(data["task"], "object-detection")
        self.assertEqual(data["runtime"]["engine"], "onnxruntime")
        self.assertIn("model", data["files"])
        self.assertIn("labels", data["files"])


if __name__ == "__main__":
    unittest.main()
