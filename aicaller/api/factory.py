from abc import ABC

from classconfig import get_configurable_attributes

from aicaller.api import APIConfigMixin, OpenAPI, OllamaAPI
from aicaller.api.api_async import OpenAsyncAPI, OllamaAsyncAPI


class APIFactory(ABC, APIConfigMixin):
    """
    Abstract factory class to create API instances.
    """

    def create(self, **kwargs):
        """
        Creates an API instance based on the provided configuration.

        :param kwargs: Configuration parameters for the API. It Will override the default API type if specified.
        :return: An instance of the API class.
        """
        ...

    def create_async(self, **kwargs):
        """
        Creates an asynchronous API instance based on the provided configuration.

        :param kwargs: Configuration parameters for the API.It Will override the default API type if specified.
        :return: An instance of the asynchronous API class.
        """
        ...

    def mixin_kwargs(self, **kwargs):
        """
        Merges the provided keyword arguments with the class configuration.

        :param kwargs: Additional configuration parameters.
        :return: A dictionary of merged configuration parameters.
        """
        configurable_attrs = get_configurable_attributes(self.__class__)
        return {**{k: getattr(self, k) for k in configurable_attrs}, **kwargs}


class OpenAPIFactory(APIFactory):
    """
    Factory class to create OpenAI API instances.
    """

    def create(self, **kwargs):
        kwargs = self.mixin_kwargs(**kwargs)
        return OpenAPI(**kwargs)

    def create_async(self, **kwargs):
        kwargs = self.mixin_kwargs(**kwargs)
        return OpenAsyncAPI(**kwargs)


class OllamaAPIFactory(APIFactory):
    """
    Factory class to create Ollama API instances.
    """

    def create(self, **kwargs):
        kwargs = self.mixin_kwargs(**kwargs)
        return OllamaAPI(**kwargs)

    def create_async(self, **kwargs):
        kwargs = self.mixin_kwargs(**kwargs)
        return OllamaAsyncAPI(**kwargs)  # Assuming OllamaClient supports async operations