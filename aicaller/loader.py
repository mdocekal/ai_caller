import json
from abc import ABC, abstractmethod
from typing import Optional

from classconfig import ConfigurableMixin, ConfigurableValue, RelativePathTransformer
from classconfig.validators import StringValidator, AnyValidator, IsNoneValidator
from datasets import Dataset, Features, Value, load_dataset, load_from_disk


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

    # JSON type -> datasets feature, for flat records
    _SCALAR_FEATURES = {
        frozenset({str}): "string",
        frozenset({bool}): "bool",
        frozenset({int}): "int64",
        frozenset({float}): "float64",
        frozenset({int, float}): "float64",
    }

    @classmethod
    def infer_features(cls, p: str) -> Optional[Features]:
        """
        Infers the features from the whole file.

        datasets infers the schema of a JSON file from its first block only, so a field that is null at the
        beginning of the file and e.g. a string later fails with "Couldn't cast array of type string to null".
        This reads every record and assigns each field the type of its non-null values.

        :param p: path to the JSONL file
        :return: features for flat records with scalar values, None (leave inference to datasets) if a field
            holds lists/objects or values of incompatible types
        """
        types = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                for k, v in json.loads(line).items():
                    types.setdefault(k, set()).add(type(v))

        features = {}
        for k, ts in types.items():
            ts.discard(type(None))
            if not ts:
                features[k] = Value("null")
                continue
            dtype = cls._SCALAR_FEATURES.get(frozenset(ts))
            if dtype is None:
                return None
            features[k] = Value(dtype)
        return Features(features)

    def _load(self, p: str) -> Dataset:
        return load_dataset("json", data_files=p, features=self.infer_features(p))["train"]


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

    load_from_disk: bool = ConfigurableValue(
        "Uses the load_from_disk method instead of load_dataset. This is useful for loading already processed datasets that are saved to disk.",
        user_default=False
    )

    def _load(self, p: str) -> Dataset:
        if self.load_from_disk:
            return load_from_disk(p)
        return load_dataset(p, self.config, split=self.split)


class HFImageLoader(Loader):
    """
    Loader for Hugging Face image datasets.
    """

    def _load(self, p: str) -> Dataset:
        return load_dataset("imagefolder", data_dir=p, split=self.split)
