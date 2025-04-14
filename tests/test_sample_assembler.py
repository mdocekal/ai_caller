from pathlib import Path
from unittest import TestCase, mock

from datasets import load_dataset

from aicaller.few_shot_sampler import FewShotSampler
from aicaller.loader import HFLoader
from aicaller.sample_assembler import TextDatasetAssembler, ImageDatasetAssembler
from aicaller.template import StringTemplate

SCRIPT_PATH = Path(__file__).parent
FIXTURES_PATH = SCRIPT_PATH / "fixtures"


class TestTextDatasetAssembler(TestCase):
    def setUp(self):
        self.dataset = load_dataset("json", data_files=str(FIXTURES_PATH / "dataset.jsonl"))["train"]

    def test_assemble(self):
        assembler = TextDatasetAssembler(StringTemplate("This is the text of {{text}}."))

        samples = list(assembler.assemble(self.dataset))

        self.assertEqual(10, len(samples))
        self.assertEqual("This is the text of 0. sample.", samples[0][0])
        self.assertEqual("This is the text of 5. sample.", samples[5][0])
        self.assertEqual("This is the text of 9. sample.", samples[9][0])

    def test_assemble_direct(self):
        assembler = TextDatasetAssembler(StringTemplate("This is the text of {{text}}."), direct="text")

        samples = list(assembler.assemble(self.dataset))

        self.assertEqual(10, len(samples))
        self.assertEqual("0. sample", samples[0][0])
        self.assertEqual("5. sample", samples[5][0])
        self.assertEqual("9. sample", samples[9][0])

    def test_assembler_select(self):
        assembler = TextDatasetAssembler(StringTemplate("This is the text of {{text}}."))

        samples = list(assembler.assemble(self.dataset, select=[0, 5, 9]))
        self.assertEqual(3, len(samples))
        self.assertEqual("This is the text of 0. sample.", samples[0][0])
        self.assertEqual("This is the text of 5. sample.", samples[1][0])
        self.assertEqual("This is the text of 9. sample.", samples[2][0])

    @mock.patch("aicaller.few_shot_sampler.random.sample")
    def test_assembler_few_shot(self, mock_random_sample):
        mock_random_sample.return_value = [0, 5, 9]
        loader = HFLoader(
            path_to=str(FIXTURES_PATH / "dataset"),
            config="short",
            split="train",
        )
        template = """There are three samples:
{% for sample in few_shot %}{{sample['text']}}
{% endfor %}
"""
        assembler = TextDatasetAssembler(
            StringTemplate(template),
            few_shot_sampler=FewShotSampler(
                load=loader,
                n=3
            )
        )

        samples = list(assembler.assemble(self.dataset))
        self.assertEqual(10, len(samples))
        self.assertEqual("There are three samples:\n0. train sample in short config\n5. train sample in short config\n9. train sample in short config\n", samples[0][0])
        self.assertEqual("There are three samples:\n0. train sample in short config\n5. train sample in short config\n9. train sample in short config\n", samples[5][0])
        self.assertEqual("There are three samples:\n0. train sample in short config\n5. train sample in short config\n9. train sample in short config\n", samples[9][0])


class TestImageDatasetAssembler(TestCase):
    def setUp(self):
        self.dataset = load_dataset("imagefolder", data_dir=str(FIXTURES_PATH / "dataset_images"), split="train")

    def test_assemble(self):
        assembler = ImageDatasetAssembler(StringTemplate("This is the image of {{file_name}}."))

        samples = list(assembler.assemble(self.dataset))

        self.assertEqual(3, len(samples))
        self.assertEqual("train_0", samples[0][1]["file_name"])
        self.assertEqual("train_1", samples[1][1]["file_name"])
        self.assertEqual("train_2", samples[2][1]["file_name"])

    def test_assemble_select(self):
        assembler = ImageDatasetAssembler(StringTemplate("This is the image of {{file_name}}."))

        samples = list(assembler.assemble(self.dataset, select=[1]))
        self.assertEqual(1, len(samples))
        self.assertEqual("train_1", samples[0][1]["file_name"])
