import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from datasets import Features, Value

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

    def write_jsonl(self, records: list) -> str:
        path = Path(self.tmp_dir.name) / "data.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(path)

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_infer_features_scalars(self):
        p = self.write_jsonl([
            {"s": "a", "i": 1, "f": 1.5, "b": True, "n": None, "if": 1, "sn": None},
            {"s": "b", "i": 2, "f": 2.5, "b": False, "n": None, "if": 2.5, "sn": "x"},
        ])

        features = JSONLLoader.infer_features(p)

        self.assertEqual(Features({
            "s": Value("string"),
            "i": Value("int64"),
            "f": Value("float64"),
            "b": Value("bool"),
            "n": Value("null"),
            "if": Value("float64"),
            "sn": Value("string"),
        }), features)

    def test_infer_features_field_missing_in_some_records(self):
        p = self.write_jsonl([{"a": "x"}, {"a": "y", "b": 3}])

        self.assertEqual(Features({"a": Value("string"), "b": Value("int64")}), JSONLLoader.infer_features(p))

    def test_infer_features_skips_empty_lines(self):
        path = Path(self.tmp_dir.name) / "data.jsonl"
        path.write_text('{"a": "x"}\n\n   \n{"a": "y"}\n', encoding="utf-8")

        self.assertEqual(Features({"a": Value("string")}), JSONLLoader.infer_features(str(path)))

    def test_infer_features_nested_falls_back(self):
        self.assertIsNone(JSONLLoader.infer_features(self.write_jsonl([{"a": [1, 2]}])))
        self.assertIsNone(JSONLLoader.infer_features(self.write_jsonl([{"a": {"b": 1}}])))

    def test_infer_features_incompatible_types_fall_back(self):
        self.assertIsNone(JSONLLoader.infer_features(self.write_jsonl([{"a": "x"}, {"a": 1}])))
        self.assertIsNone(JSONLLoader.infer_features(self.write_jsonl([{"a": True}, {"a": 1}])))

    def test_load_null_then_string_beyond_first_block(self):
        # datasets infers the schema from the first block (~10MB) only; a field that is null there and a
        # string later used to fail with "Couldn't cast array of type string to null"
        pad = "x" * 5000
        records = [{"id": i, "text": pad, "thinking": None if i < 4000 else f"thought {i}"} for i in range(6000)]
        loader = JSONLLoader(path_to=self.write_jsonl(records))

        dataset = loader.load()

        self.assertEqual(6000, len(dataset))
        self.assertIsNone(dataset[0]["thinking"])
        self.assertIsNone(dataset[3999]["thinking"])
        self.assertEqual("thought 4000", dataset[4000]["thinking"])
        self.assertEqual("thought 5999", dataset[5999]["thinking"])
        self.assertEqual(Value("string"), dataset.features["thinking"])

    def test_load_nested_still_loads(self):
        loader = JSONLLoader(path_to=self.write_jsonl([{"id": 0, "tags": ["a", "b"]}, {"id": 1, "tags": ["c"]}]))

        dataset = loader.load()

        self.assertEqual(2, len(dataset))
        self.assertEqual(["a", "b"], dataset[0]["tags"])


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

