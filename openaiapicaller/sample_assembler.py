import json
from abc import ABC, abstractmethod

import json
from abc import ABC
from typing import Generator, Union, Optional

import jinja2
from classconfig import ConfigurableValue, RelativePathTransformer
from classconfig.validators import StringValidator
from datasets import load_dataset


class APISampleAssembler(ABC):
    """
    Base class for assemblers that are used to create samples for API requests.
    """

    @abstractmethod
    def assemble(self, p: Optional[str]) -> Generator[tuple[str, dict[str, Union[str, int]]], None, None]:
        """
        Assembles samples for API requests.

        :param p: Path to data.
        :return: Generator of assembled samples.
            In form of tuple:
                sample
                sample ids for construction of request id
        """
        ...


class Jinja2Assembler(APISampleAssembler, ABC):
    """
    Jinja2 template based sample assembler.
    """

    input_template: str = ConfigurableValue("Jinja2 template for input assembly. You can use fields from the input data.",
                                            validator=StringValidator())

    def __init__(self, input_template: str):
        """
        Initializes the assembler.

        :param input_template: Jinja2 template for input assembly. You can use fields from the input data.
        """
        self.input_template = input_template
        self.jinja = jinja2.Environment()
        self.jinja_input_template = self.jinja.from_string(self.input_template)


class JSONLAssembler(Jinja2Assembler):
    """
    Assembles samples from jsonl file.
    """

    def assemble(self, p: str) -> Generator[tuple[str, dict[str, Union[str, int]]], None, None]:
        """
        Assembles samples from search results.

        :param p: path to jsonl file with search results
        :return: Generator of assembled samples.
            In form of tuple:
                sample
                dictionary of identifiers
                    {
                        line_number: number of the line (starting from 0)
                    }
        """

        with open(p, mode='r') as f:
            for line_number, line in enumerate(f):
                data = json.loads(line)
                sample = self.jinja_input_template.render(data)
                yield sample, {"line_number": line_number}


class HuggingfaceDatasetAssembler(Jinja2Assembler):
    """
    Assembles samples from Huggingface dataset.
    """

    dataset: str = ConfigurableValue("Name or path of the dataset.", validator=StringValidator(), voluntary=True,
                                     transform=RelativePathTransformer(force_relative_prefix=True))
    split: str = ConfigurableValue("Split of the dataset.", validator=StringValidator())
    config: Optional[str] = ConfigurableValue("Configuration name.", voluntary=True, validator=StringValidator())

    def __init__(self, input_template: str, dataset: str, split: str, config: Optional[str] = None):
        """
        Initializes the assembler.

        :param input_template: Jinja2 template for input assembly. You can use fields from the input data.
        :param dataset: Name or path of the dataset.
        :param split: Split of the dataset.
        :param config: Configuration name.
        """
        super().__init__(input_template)
        self.dataset = dataset
        self.split = split
        self.config = config

    def assemble(self, p: Optional[str]) -> Generator[tuple[str, dict[str, Union[str, int]]], None, None]:
        """
        Assembles samples from Huggingface dataset.

        :param p: Path to the dataset.
        :return: Generator of assembled samples.
            In form of tuple:
                sample
                dictionary of identifiers
                    {
                        line_number: number of the line (starting from 0)
                    }
        """

        dataset = load_dataset(self.dataset, self.config)
        for i, sample in enumerate(dataset[self.split]):
            yield self.jinja_input_template.render(sample), {"line_number": i}
