from abc import ABC, abstractmethod
from typing import Optional

from classconfig import ConfigurableMixin, ConfigurableValue, RelativePathTransformer
from classconfig.validators import StringValidator, AnyValidator, IsNoneValidator
from datasets import Dataset, load_dataset


class Loader(ABC, ConfigurableMixin):
    """
    Base class for loaders.
    """

    path_to: str = ConfigurableValue(
        "Path to the data.",
        transform=RelativePathTransformer(force_relative_prefix=True)
    )
    config: Optional[str] = ConfigurableValue("Configuration name.", voluntary=True, validator=AnyValidator([IsNoneValidator(), StringValidator()]))
    split: Optional[str] = ConfigurableValue("Split of the dataset.", voluntary=True, validator=AnyValidator([IsNoneValidator(), StringValidator()]))

    @abstractmethod
    def _load(self, p: str) -> Dataset:
        """
        Loads the dataset.

        :param p: path to the data
        :return: Loaded dataset.
        """
        ...

    def load(self, p: Optional[str] = None) -> Dataset:
        """
        Loads the dataset.

        :param p: Voluntary path to the data. If not provided, the path from the configuration is used.
        :return: Loaded dataset.
        """
        return self._load(self.path_to if p is None else p)


class JSONLLoader(Loader):
    """
    Loader for JSONL files.
    """

    def _load(self, p: str) -> Dataset:
        return load_dataset("json", data_files=p)["train"]


class CSVLoader(Loader):
    """
    Loader for CSV files.
    """

    def _load(self, p: str) -> Dataset:
        return load_dataset("csv", data_files=p)["train"]


class HFLoader(Loader):
    """
    Loader for Hugging Face datasets.
    """

    def _load(self, p: str) -> Dataset:
        return load_dataset(p, self.config, split=self.split)


class HFImageLoader(Loader):
    """
    Loader for Hugging Face image datasets.
    """

    def _load(self, p: str) -> Dataset:
        return load_dataset("imagefolder", data_dir=p, split=self.split)
