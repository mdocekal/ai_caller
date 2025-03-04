from pathlib import Path
from unittest import TestCase, mock

from openaiapicaller.few_shot_sampler import FewShotSampler
from openaiapicaller.loader import JSONLLoader

SCRIPT_PATH = Path(__file__).parent
FIXTURES_PATH = SCRIPT_PATH / "fixtures"


class TestFewShotSampler(TestCase):

    def setUp(self):
        self.loader = JSONLLoader(path_to=str(FIXTURES_PATH / "dataset.jsonl"))

    @mock.patch("openaiapicaller.few_shot_sampler.random.sample")
    def test_3_shot(self, mock_random_sample):
        mock_random_sample.return_value = [0, 5, 9]

        sampler = FewShotSampler(load=self.loader)

        indices, samples = sampler.sample(3)

        self.assertEqual(3, len(samples))
        self.assertSequenceEqual([0, 5, 9], indices)
        self.assertDictEqual({"id": 0, "text": "0. sample"}, samples[0])
        self.assertDictEqual({"id": 5, "text": "5. sample"}, samples[1])
        self.assertDictEqual({"id": 9, "text": "9. sample"}, samples[2])
