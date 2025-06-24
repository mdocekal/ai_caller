from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import Mock, MagicMock

from aicaller.few_shot_sampler import FewShotSampler
from aicaller.loader import JSONLLoader

SCRIPT_PATH = Path(__file__).parent
FIXTURES_PATH = SCRIPT_PATH / "fixtures"


class TestFewShotSampler(TestCase):

    def setUp(self):
        self.loader = JSONLLoader(path_to=str(FIXTURES_PATH / "dataset.jsonl"))

    def test_3_shot(self):

        sampler = FewShotSampler(load=self.loader)
        sampler.r = MagicMock()
        sampler.r.sample.return_value = [0, 5, 9]

        indices, samples = sampler.sample(3)

        self.assertEqual(3, len(samples))
        self.assertSequenceEqual([0, 5, 9], indices)
        self.assertDictEqual({"id": 0, "text": "0. sample"}, samples[0])
        self.assertDictEqual({"id": 5, "text": "5. sample"}, samples[1])
        self.assertDictEqual({"id": 9, "text": "9. sample"}, samples[2])
