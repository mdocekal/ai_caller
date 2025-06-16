from pathlib import Path
from unittest import TestCase

from aicaller.loader import JSONLLoader, CSVLoader, HFLoader, HFImageLoader

SCRIPT_PATH = Path(__file__).parent.resolve()
FIXTURES_PATH = SCRIPT_PATH / "fixtures"


class TestJSONLoader(TestCase):

    def test_load(self):
        loader = JSONLLoader(path_to=str(FIXTURES_PATH / "dataset.jsonl"))

        dataset = loader.load()

        self.assertEqual(10, len(dataset))
        self.assertDictEqual({"id": 0, "text": "0. sample"}, dataset[0])
        self.assertDictEqual({"id": 5, "text": "5. sample"}, dataset[5])
        self.assertDictEqual({"id": 9, "text": "9. sample"}, dataset[9])

    def test_load_override(self):
        loader = JSONLLoader(path_to=str(FIXTURES_PATH / "dataset_non_existent.jsonl"))

        dataset = loader.load(str(FIXTURES_PATH / "dataset.jsonl"))

        self.assertEqual(10, len(dataset))
        self.assertDictEqual({"id": 0, "text": "0. sample"}, dataset[0])
        self.assertDictEqual({"id": 5, "text": "5. sample"}, dataset[5])
        self.assertDictEqual({"id": 9, "text": "9. sample"}, dataset[9])


class TestCSVLoader(TestCase):

    def test_load(self):
        loader = CSVLoader(path_to=str(FIXTURES_PATH / "dataset.csv"))

        dataset = loader.load()

        self.assertEqual(10, len(dataset))
        self.assertDictEqual({"id": 0, "text": "0. sample"}, dataset[0])
        self.assertDictEqual({"id": 5, "text": "5. sample"}, dataset[5])
        self.assertDictEqual({"id": 9, "text": "9. sample"}, dataset[9])

    def test_load_override(self):
        loader = CSVLoader(path_to=str(FIXTURES_PATH / "dataset_non_existent.csv"))

        dataset = loader.load(str(FIXTURES_PATH / "dataset.csv"))

        self.assertEqual(10, len(dataset))
        self.assertDictEqual({"id": 0, "text": "0. sample"}, dataset[0])
        self.assertDictEqual({"id": 5, "text": "5. sample"}, dataset[5])
        self.assertDictEqual({"id": 9, "text": "9. sample"}, dataset[9])


class TestHFLoader(TestCase):
    def test_load(self):
        loader = HFLoader(path_to=str(FIXTURES_PATH / "dataset"), config="long", split="test")

        dataset = loader.load()

        self.assertEqual(10, len(dataset))
        self.assertDictEqual({"id": 0, "text": "0. test sample in long config"}, dataset[0])
        self.assertDictEqual({"id": 5, "text": "5. test sample in long config"}, dataset[5])
        self.assertDictEqual({"id": 9, "text": "9. test sample in long config"}, dataset[9])

    def test_load_override(self):
        loader = HFLoader(path_to=str(FIXTURES_PATH / "dataset_non_existent"), config="long", split="test")

        dataset = loader.load(str(FIXTURES_PATH / "dataset"))

        self.assertEqual(10, len(dataset))
        self.assertDictEqual({"id": 0, "text": "0. test sample in long config"}, dataset[0])
        self.assertDictEqual({"id": 5, "text": "5. test sample in long config"}, dataset[5])
        self.assertDictEqual({"id": 9, "text": "9. test sample in long config"}, dataset[9])


class TestHFImageLoader(TestCase):
    def test_load(self):
        loader = HFImageLoader(path_to=str(FIXTURES_PATH / "dataset_images"), split="test")

        dataset = loader.load()

        self.assertEqual(3, len(dataset))
        self.assertEqual(str(FIXTURES_PATH / "dataset_images" / "test" / "test_0.jpg"), dataset[0]["image"].filename)
        self.assertEqual(str(FIXTURES_PATH / "dataset_images" / "test" / "test_1.jpg"), dataset[1]["image"].filename)

    def test_load_override(self):
        loader = HFImageLoader(path_to=str(FIXTURES_PATH / "dataset_non_existent"), split="test")

        dataset = loader.load(str(FIXTURES_PATH / "dataset_images"))

        self.assertEqual(3, len(dataset))
        self.assertEqual(str(FIXTURES_PATH / "dataset_images" / "test" / "test_0.jpg"), dataset[0]["image"].filename)
        self.assertEqual(str(FIXTURES_PATH / "dataset_images" / "test" / "test_1.jpg"), dataset[1]["image"].filename)

