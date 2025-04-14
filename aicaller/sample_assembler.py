from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Generator, Union, Optional, Any, Sequence

from classconfig import ConfigurableValue, ConfigurableSubclassFactory, ConfigurableFactory
from classconfig.validators import StringValidator, AnyValidator, IsNoneValidator, ListOfTypesValidator
from datasets import Dataset

from aicaller.few_shot_sampler import FewShotSampler
from aicaller.template import StringTemplate, Template


class APISampleAssembler(ABC):
    """
    Base class for assemblers that are used to create samples for API requests.
    """
    few_shot_sampler: Optional[FewShotSampler] = ConfigurableFactory(
        FewShotSampler,
        "Few shot sampler for sampling examples. It will save few-shot indices and samples to few_shot_indices and few_shot fields.",
        voluntary=True
    )

    id_fields: Optional[Sequence[str]] = ConfigurableValue(
        "List of additional dataset fields to use for sample ids.",
        user_default=None,
        voluntary=True,
        validator=AnyValidator([IsNoneValidator(), ListOfTypesValidator(str, allow_empty=True)]),
    )

    def __init__(self, few_shot_sampler: Optional[FewShotSampler], id_fields: Optional[Sequence[str]] = None):
        self.few_shot_sampler = few_shot_sampler
        self.id_fields = id_fields

    @abstractmethod
    def assemble(self, dataset: Dataset, select: Optional[Sequence[int]] = None) -> Generator[tuple[Any, dict[str, Union[str, int]]], None, None]:
        """
        Assembles samples for API requests.

        :param dataset: Dataset for assembly of samples.
        :param select: List of indices to select. Else all samples are selected.
        :return: Generator of assembled samples.
            In form of tuple:
                sample
                sample ids for construction of request id
        """
        ...


class TemplateBasedAssembler(APISampleAssembler, ABC):
    """
    Template based sample assembler.
    """

    input_template: Template = ConfigurableSubclassFactory(Template, "Template for input assembly.",
                                                           user_default=StringTemplate)

    def __init__(self, input_template: Template, few_shot_sampler: Optional[FewShotSampler] = None,
                 id_fields: Optional[Sequence[str]] = None):
        """
        Initializes the assembler.

        """
        super().__init__(few_shot_sampler, id_fields)
        self.input_template = input_template

    def add_few_shot(self, sample: dict):
        """
        Adds few-shot indices and samples to the sample.

        :param sample: Sample to add few-shot to.
        """
        if self.few_shot_sampler is not None:
            sample["few_shot_indices"], sample["few_shot"] = self.few_shot_sampler.sample()


class TextDatasetAssembler(TemplateBasedAssembler):
    """
    Assembles samples from text dataset.
    """

    direct: Optional[str] = ConfigurableValue("Name of jsonl field that contains the sample. In that case, the template is not used.",
                                                voluntary=True, validator=AnyValidator([IsNoneValidator(), StringValidator()]))

    def __init__(self, input_template: Template, few_shot_sampler: Optional[FewShotSampler] = None,
                 id_fields: Optional[Sequence[str]] = None, direct: Optional[str] = None):
        """
        Initializes the assembler.

        """
        super().__init__(
            few_shot_sampler=few_shot_sampler,
            input_template=input_template,
            id_fields=id_fields
        )
        self.direct = direct

    def assemble(self, dataset: Dataset, select: Optional[Sequence[int]] = None) -> Generator[tuple[Any, dict[str, Union[str, int]]], None, None]:
        """
        Assembles samples from search results.

        :param dataset: Dataset for assembly of samples.
        :param select: List of indices to select. Else all samples are selected.
        :return: Generator of assembled samples.
            In form of tuple:
                sample
                dictionary of identifiers
                    {
                        line_number: number of the line (starting from 0)
                    }
        """

        for line_number, sample in enumerate(dataset):
            sample_ids = {"line_number": line_number}
            if self.id_fields is not None:
                for field in self.id_fields:
                    sample_ids[field] = sample[field]

            if select is not None and line_number not in select:
                continue
            if self.direct:
                sample = sample[self.direct]
            else:
                self.add_few_shot(sample)
                sample = self.input_template.render(sample)

            yield sample, sample_ids


class ImageDatasetAssembler(TemplateBasedAssembler):
    def __init__(self, input_template: Template, few_shot_sampler: Optional[FewShotSampler] = None,
                 id_fields: Optional[Sequence[str]] = None):

        super().__init__(
            few_shot_sampler=few_shot_sampler,
            input_template=input_template,
            id_fields=id_fields
        )

    def assemble(self, dataset: Dataset, select: Optional[Sequence[int]] = None) -> Generator[tuple[Any, dict[str, Union[str, int]]], None, None]:
        """
        Assembles samples from Huggingface image folder dataset.

        :param dataset: Dataset for assembly of samples.
        :param select: List of indices to select. Else all samples are selected.
        :return: Generator of assembled samples.
            In form of tuple:
                sample
                dictionary of identifiers
                    {
                        image_path: path to the image
                        file_name: name of the file (without extension)
                    }
        """
        if select:
            dataset = dataset.select(select)

        for sample in dataset:
            sample_ids = {
                "image_path": sample["image"].filename,
                "file_name": Path(sample["image"].filename).stem
            }
            if self.id_fields is not None:
                for field in self.id_fields:
                    sample_ids[field] = sample[field]

            self.add_few_shot(sample)
            yield self.input_template.render(sample), sample_ids
