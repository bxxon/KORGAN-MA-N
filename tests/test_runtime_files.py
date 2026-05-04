import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeFilesTest(unittest.TestCase):
    def test_runtime_config_threshold_order(self) -> None:
        config_path = REPO_ROOT / "config" / "runtime_config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        detection = data["detection"]
        tracking = data["tracking"]
        self.assertLessEqual(detection["detector_conf_threshold"], detection["tracker_low_conf_threshold"])
        self.assertLessEqual(detection["tracker_low_conf_threshold"], detection["tracker_high_conf_threshold"])
        self.assertLessEqual(detection["tracker_high_conf_threshold"], 1.0)
        self.assertLessEqual(detection["nms_threshold"], 1.0)
        self.assertGreaterEqual(tracking["kilit_aralik_bonus"], 0.0)
        self.assertLessEqual(tracking["kilit_aralik_bonus"], 1.0)

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

    def test_model_package_files_exist(self) -> None:
        required_files = ("model.onnx", "labels.txt", "model-manifest.json")
        package_dirs = (
            REPO_ROOT / "deliverables",
            REPO_ROOT / "deliverables" / "onnxruntime_cpu_package",
        )

        for package_dir in package_dirs:
            for filename in required_files:
                file_path = package_dir / filename
                self.assertTrue(file_path.exists(), f"Missing runtime file: {file_path}")

    def test_mission_target_example_profiles_are_valid(self) -> None:
        config_path = REPO_ROOT / "config" / "runtime_config_mission_targets.example.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        profiles = data["hedef_profilleri"]
        self.assertGreaterEqual(len(profiles), 3)

        seen_names: set[str] = set()
        seen_labels: set[str] = set()
        for profile in profiles:
            self.assertTrue(profile["ad"])
            self.assertTrue(profile["etiket"])
            self.assertGreater(profile["gercek_genislik_cm"], 0.0)
            self.assertGreaterEqual(profile["min_kilit_mesafe_m"], 0.0)
            self.assertLessEqual(profile["min_kilit_mesafe_m"], profile["max_kilit_mesafe_m"])
            self.assertGreater(profile["oncelik"], 0)
            self.assertNotIn(profile["ad"], seen_names)
            self.assertNotIn(profile["etiket"], seen_labels)
            seen_names.add(profile["ad"])
            seen_labels.add(profile["etiket"])

        tracking = data["tracking"]
        self.assertEqual(tracking["kilit_icin_mesafe_zorunlu"], 1)
        self.assertGreater(tracking["kilit_aralik_bonus"], 0.0)


if __name__ == "__main__":
    unittest.main()
