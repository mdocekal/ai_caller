# -*- coding: UTF-8 -*-
"""
Created on 17.02.25

:author:     Martin Dočekal
"""
import unittest
from pathlib import Path
from unittest import TestCase
from aicaller.template import StringTemplate, SegmentedStringTemplate, OpenAIMessageBuilder, OpenAITextContent, \
    OpenAIImageContent, OpenAIMultiModalMessageBuilder, OllamaMessageBuilder, MessagesTemplate, \
    Jinja2EnvironmentSingletonFactory, truncate_by_tokens, HAS_TIKTOKEN, HAS_TRANSFORMERS

SCRIPT_PATH = Path(__file__).parent
FIXTURES_PATH = SCRIPT_PATH / "fixtures"


class TestStringTemplate(TestCase):

    def test_render(self):
        template = StringTemplate("Hello {{name}}!")
        self.assertEqual("Hello Alan!", template.render({"name": "Alan"}))


class TestSegmentedStringTemplate(TestCase):

    def test_render(self):
        template = SegmentedStringTemplate({
            "start": "Hello ",
            "name": "{{name}}!"
        })
        r = template.render({"name": "Alan"})
        self.assertEqual("Hello Alan!", r)
        self.assertSequenceEqual(["start", "name"], r.labels)
        self.assertSequenceEqual(["Hello ", "Alan!"], r.segments)


class TestOpenAIMessageBuilder(TestCase):

    def test_render(self):
        msg_builder = OpenAIMessageBuilder(
            role="user",
            content="Hello {{name}}!"
        )

        self.assertEqual({"role": "user", "content": "Hello Alan!"}, msg_builder.render({"name": "Alan"}))

        msg_builder = OpenAIMessageBuilder(
            role="system",
            content="Hello {{name}}!"
        )

        self.assertEqual({"role": "system", "content": "Hello Alan!"}, msg_builder.render({"name": "Alan"}))


class TestOpenAITextContent(TestCase):

    def test_render(self):
        msg_builder = OpenAITextContent(
            text="Hello {{name}}!"
        )

        self.assertEqual({"type": "text", "text": "Hello Alan!"}, msg_builder.render({"name": "Alan"}))


PIXEL_PNG_PATH = FIXTURES_PATH / "pixel.png"
PIXEL_JPG_PATH = FIXTURES_PATH / "pixel.jpg"


class TestOpenAIImageContent(TestCase):

    def test_render(self):
        content = OpenAIImageContent(
            url="{{filename}}"
        )

        res = content.render({"filename": str(PIXEL_PNG_PATH)})
        self.assertEqual("image_url", res["type"])
        self.assertEqual("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TpaVUROwgIpKhOlkQFXHUKhShQqgVWnUwufQLmjQkLS6OgmvBwY/FqoOLs64OroIg+AHiLjgpukiJ/0sKLWI8OO7Hu3uPu3eA0CgzzeoaBzS9aqYScTGTXRUDrxDQhyBCGJaZZcxJUhKe4+sePr7exXiW97k/R4+asxjgE4lnmWFWiTeIpzerBud94ggryirxOfGYSRckfuS64vIb54LDAs+MmOnUPHGEWCx0sNLBrGhqxFPEUVXTKV/IuKxy3uKslWusdU/+wnBOX1nmOs0hJLCIJUgQoaCGEsqoIkarToqFFO3HPfyDjl8il0KuEhg5FlCBBtnxg//B726t/OSEmxSOA90vtv0xAgR2gWbdtr+Pbbt5AvifgSu97a80gJlP0uttLXoE9G4DF9dtTdkDLneAgSdDNmVH8tMU8nng/Yy+KQv03wKhNbe31j5OH4A0dZW8AQ4OgdECZa97vDvY2du/Z1r9/QAeI3KFtf3sAQAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAAd0SU1FB+kCEQgVASFbBEoAAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAAADElEQVQI12NgYGAAAAAEAAEnNCcKAAAAAElFTkSuQmCC",
                         res["image_url"]["url"])

        res = content.render({"filename": str(PIXEL_JPG_PATH)})
        self.assertEqual("image_url", res["type"])
        self.assertEqual("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEBLAEsAAD//gATQ3JlYXRlZCB3aXRoIEdJTVD/4gKwSUNDX1BST0ZJTEUAAQEAAAKgbGNtcwQwAABtbnRyUkdCIFhZWiAH6QACABEACAAKAB5hY3NwQVBQTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9tYAAQAAAADTLWxjbXMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1kZXNjAAABIAAAAEBjcHJ0AAABYAAAADZ3dHB0AAABmAAAABRjaGFkAAABrAAAACxyWFlaAAAB2AAAABRiWFlaAAAB7AAAABRnWFlaAAACAAAAABRyVFJDAAACFAAAACBnVFJDAAACFAAAACBiVFJDAAACFAAAACBjaHJtAAACNAAAACRkbW5kAAACWAAAACRkbWRkAAACfAAAACRtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACQAAAAcAEcASQBNAFAAIABiAHUAaQBsAHQALQBpAG4AIABzAFIARwBCbWx1YwAAAAAAAAABAAAADGVuVVMAAAAaAAAAHABQAHUAYgBsAGkAYwAgAEQAbwBtAGEAaQBuAABYWVogAAAAAAAA9tYAAQAAAADTLXNmMzIAAAAAAAEMQgAABd7///MlAAAHkwAA/ZD///uh///9ogAAA9wAAMBuWFlaIAAAAAAAAG+gAAA49QAAA5BYWVogAAAAAAAAJJ8AAA+EAAC2xFhZWiAAAAAAAABilwAAt4cAABjZcGFyYQAAAAAAAwAAAAJmZgAA8qcAAA1ZAAAT0AAACltjaHJtAAAAAAADAAAAAKPXAABUfAAATM0AAJmaAAAmZwAAD1xtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAEcASQBNAFBtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEL/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wgARCAABAAEDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhADEAAAASof/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABCf/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxB//9k=",
                         res["image_url"]["url"])


class TestOpenAIMultiModalMessageBuilder(TestCase):

    def test_render(self):
        msg_builder = OpenAIMultiModalMessageBuilder(
            role="assistant",
            content=[
                OpenAITextContent(text="Hello {{name}}!"),
                OpenAIImageContent(url="{{filename}}")
            ]
        )

        res = msg_builder.render({"name": "Alan", "filename": str(PIXEL_PNG_PATH)})
        self.assertEqual(2, len(res))
        self.assertEqual("assistant", res["role"])
        self.assertSequenceEqual([
            {
                "type": "text",
                "text": "Hello Alan!"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TpaVUROwgIpKhOlkQFXHUKhShQqgVWnUwufQLmjQkLS6OgmvBwY/FqoOLs64OroIg+AHiLjgpukiJ/0sKLWI8OO7Hu3uPu3eA0CgzzeoaBzS9aqYScTGTXRUDrxDQhyBCGJaZZcxJUhKe4+sePr7exXiW97k/R4+asxjgE4lnmWFWiTeIpzerBud94ggryirxOfGYSRckfuS64vIb54LDAs+MmOnUPHGEWCx0sNLBrGhqxFPEUVXTKV/IuKxy3uKslWusdU/+wnBOX1nmOs0hJLCIJUgQoaCGEsqoIkarToqFFO3HPfyDjl8il0KuEhg5FlCBBtnxg//B726t/OSEmxSOA90vtv0xAgR2gWbdtr+Pbbt5AvifgSu97a80gJlP0uttLXoE9G4DF9dtTdkDLneAgSdDNmVH8tMU8nng/Yy+KQv03wKhNbe31j5OH4A0dZW8AQ4OgdECZa97vDvY2du/Z1r9/QAeI3KFtf3sAQAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAAd0SU1FB+kCEQgVASFbBEoAAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAAADElEQVQI12NgYGAAAAAEAAEnNCcKAAAAAElFTkSuQmCC",
                    "detail": "auto"
                }
            }
        ], res["content"])


class TestOllamaMessageBuilder(TestCase):

    def test_render(self):
        msg_builder = OllamaMessageBuilder(
            role="user",
            content="Hello {{name}}!"
        )

        self.assertEqual({"role": "user", "content": "Hello Alan!"}, msg_builder.render({"name": "Alan"}))

    def test_render_with_image(self):
        msg_builder = OllamaMessageBuilder(
            role="user",
            content="Hello {{name}}!",
            images=["{{filename_jpg}}", "{{filename_png}}"]
        )

        res = msg_builder.render({"name": "Alan", "filename_jpg": str(PIXEL_JPG_PATH), "filename_png": str(PIXEL_PNG_PATH)})

        self.assertEqual("user", res["role"])
        self.assertEqual("Hello Alan!", res["content"])
        self.assertSequenceEqual([str(PIXEL_JPG_PATH), str(PIXEL_PNG_PATH)], res["images"])


class TestMessagesTemplate(TestCase):

    def test_render(self):
        template = MessagesTemplate([
            OpenAIMessageBuilder(
                role="system",
                content="You are {{system}}!"
            ),
            OpenAIMultiModalMessageBuilder(
                role="user",
                content=[
                    OpenAITextContent(text="Hello {{assistant}}!"),
                    OpenAIImageContent(url="{{filename}}")
                ]
            ),
        ])

        res = template.render({"system": "awesome", "assistant": "Alan", "filename": str(PIXEL_PNG_PATH)})

        self.assertSequenceEqual([
            {
                "role": "system",
                "content": "You are awesome!"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello Alan!"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TpaVUROwgIpKhOlkQFXHUKhShQqgVWnUwufQLmjQkLS6OgmvBwY/FqoOLs64OroIg+AHiLjgpukiJ/0sKLWI8OO7Hu3uPu3eA0CgzzeoaBzS9aqYScTGTXRUDrxDQhyBCGJaZZcxJUhKe4+sePr7exXiW97k/R4+asxjgE4lnmWFWiTeIpzerBud94ggryirxOfGYSRckfuS64vIb54LDAs+MmOnUPHGEWCx0sNLBrGhqxFPEUVXTKV/IuKxy3uKslWusdU/+wnBOX1nmOs0hJLCIJUgQoaCGEsqoIkarToqFFO3HPfyDjl8il0KuEhg5FlCBBtnxg//B726t/OSEmxSOA90vtv0xAgR2gWbdtr+Pbbt5AvifgSu97a80gJlP0uttLXoE9G4DF9dtTdkDLneAgSdDNmVH8tMU8nng/Yy+KQv03wKhNbe31j5OH4A0dZW8AQ4OgdECZa97vDvY2du/Z1r9/QAeI3KFtf3sAQAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAAd0SU1FB+kCEQgVASFbBEoAAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAAADElEQVQI12NgYGAAAAAEAAEnNCcKAAAAAElFTkSuQmCC",
                            "detail": "auto"
                        }
                    }
                ]
            }
        ], res)


class TestTruncateByTokensFilter(unittest.TestCase):

    def setUp(self):
        """Set up standard test data used across multiple tests."""
        # "One two three four five." is tokenized consistently by most subword tokenizers
        self.sample_text = "One two three four five."
        self.hf_model = "gpt2"
        self.tiktoken_model = "gpt-4"

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers library not installed")
    def test_transformers_truncate_right(self):
        """Test keeping the first N tokens using Transformers."""
        # Expected tokens: ['One', ' two', ' three']
        result = truncate_by_tokens(
            self.sample_text, self.hf_model, 3, direction="right", backend="transformers"
        )
        self.assertEqual(result, "One two three")

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers library not installed")
    def test_transformers_truncate_left(self):
        """Test keeping the last N tokens using Transformers."""
        # Expected tokens: [' four', ' five', '.']
        result = truncate_by_tokens(
            self.sample_text, self.hf_model, 3, direction="left", backend="transformers"
        )
        self.assertEqual(result, " four five.")

    @unittest.skipUnless(HAS_TIKTOKEN, "tiktoken library not installed")
    def test_tiktoken_truncate_right(self):
        """Test keeping the first N tokens using Tiktoken."""
        result = truncate_by_tokens(
            self.sample_text, self.tiktoken_model, 3, direction="right", backend="tiktoken"
        )
        self.assertEqual(result, "One two three")

    @unittest.skipUnless(HAS_TIKTOKEN, "tiktoken library not installed")
    def test_tiktoken_truncate_left(self):
        """Test keeping the last N tokens using Tiktoken."""
        result = truncate_by_tokens(
            self.sample_text, self.tiktoken_model, 3, direction="left", backend="tiktoken"
        )
        self.assertEqual(result, " four five.")

    def test_short_text_returns_unchanged(self):
        """Test that text shorter than the token limit returns the exact original text."""
        # Assuming transformers is available for this test
        if HAS_TRANSFORMERS:
            result = truncate_by_tokens(
                self.sample_text, self.hf_model, 100, backend="transformers"
            )
            self.assertEqual(result, self.sample_text)

    def test_empty_string(self):
        """Test that an empty string returns an empty string."""
        if HAS_TRANSFORMERS:
            self.assertEqual(truncate_by_tokens("", self.hf_model, 5), "")
        if HAS_TIKTOKEN:
            self.assertEqual(truncate_by_tokens("", self.tiktoken_model, 5, backend="tiktoken"), "")

    def test_invalid_direction(self):
        """Test that an invalid direction raises a ValueError."""
        if HAS_TRANSFORMERS:
            with self.assertRaises(ValueError):
                truncate_by_tokens(self.sample_text, self.hf_model, 3, direction="middle")

    def test_invalid_backend(self):
        """Test that an unsupported backend raises a ValueError."""
        with self.assertRaises(ValueError):
            truncate_by_tokens(self.sample_text, self.hf_model, 3, backend="spacy")


class TestJinja2EnvironmentIntegration(unittest.TestCase):

    def setUp(self):
        """Initialize the factory and get the Jinja2 environment."""
        self.factory = Jinja2EnvironmentSingletonFactory()
        self.env = self.factory.jinja_env

    def test_filter_registered(self):
        """Test that the filter is successfully registered in the Jinja environment."""
        self.assertIn("truncate", self.env.filters)

    @unittest.skipUnless(HAS_TIKTOKEN, "tiktoken library not installed")
    def test_template_rendering(self):
        """Test rendering a Jinja2 template using the custom filter."""
        template_str = '{{ text | truncate(tokenizer_name="gpt-4", number_of_tokens=2, direction="right", backend="tiktoken") }}'
        template = self.env.from_string(template_str)

        rendered = template.render(text="Alpha beta gamma delta.")
        # "Alpha", " beta"
        self.assertEqual(rendered, "Alpha beta")
