import random
from typing import Optional

from classconfig import ConfigurableSubclassFactory, ConfigurableValue

from aicaller.loader import Loader


class FewShotSampler:
    """
    Class to sample few-shot examples.
    """

    load: Loader = ConfigurableSubclassFactory(Loader, "Loader for the data.")
    n: int = ConfigurableValue("Number of examples to sample.", user_default=3)

    def __init__(self, load: Loader, n: int = 3):
        self.load = load
        self.n = n
        self.dataset = self.load.load()

    def sample(self, n: Optional[int] = None) -> tuple[list[int], list[dict]]:
        """
        Samples n examples.

        :param n: Number of examples to sample. If not provided, the number from the configuration is used.
        :return: List of examples in form of tuple:
            data, custom id fields
        """
        if n is None:
            n = self.n
        selected_indices = random.sample(list(range(len(self.dataset))), n)

        return selected_indices, [self.dataset[i] for i in selected_indices]

