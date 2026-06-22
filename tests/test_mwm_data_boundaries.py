from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ModuleNotFoundError:
    HAS_DATA_DEPS = False
else:
    HAS_DATA_DEPS = True


class MWMDataBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(HAS_DATA_DEPS, "requires numpy and torch")
    def test_data_helpers_live_in_canonical_modules(self) -> None:
        from mwm.data.metadata import dataset_metadata_path, load_dataset_metadata, write_dataset_metadata
        from mwm.data.sampling import StartGoalPair, sample_start_goal_pairs

        self.assertTrue(callable(dataset_metadata_path))
        self.assertTrue(callable(load_dataset_metadata))
        self.assertTrue(callable(write_dataset_metadata))
        self.assertTrue(callable(sample_start_goal_pairs))
        self.assertEqual(StartGoalPair.__name__, "StartGoalPair")

    @unittest.skipUnless(HAS_DATA_DEPS, "requires numpy and torch")
    def test_transform_helpers_have_canonical_module(self) -> None:
        from mwm.data.transforms import MWMTrainSampleTransform, ZScoreScaler
        from mwm.preprocessing.images import stable_pretraining_image_transforms

        self.assertEqual(MWMTrainSampleTransform.__name__, "MWMTrainSampleTransform")
        self.assertEqual(ZScoreScaler.__name__, "ZScoreScaler")
        self.assertTrue(callable(stable_pretraining_image_transforms))

    def test_image_preprocessing_implementation_lives_in_preprocessing_package(self) -> None:
        eval_policy = (ROOT / "mwm" / "eval" / "policy.py").read_text(encoding="utf-8")
        world_model = (ROOT / "mwm" / "models" / "world_model.py").read_text(encoding="utf-8")
        preprocessing = (ROOT / "mwm" / "preprocessing" / "images.py").read_text(encoding="utf-8")

        self.assertNotIn("def mwm_image_input_transform(", eval_policy)
        self.assertNotIn("def imagenet_image_input_transform(", eval_policy)
        self.assertNotIn("class ImageNetPreprocess", world_model)
        self.assertIn("def mwm_image_input_transform(", preprocessing)
        self.assertIn("def imagenet_image_input_transform(", preprocessing)
        self.assertIn("class ImageNetPreprocess", preprocessing)


if __name__ == "__main__":
    unittest.main()
