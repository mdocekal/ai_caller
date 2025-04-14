# -*- coding: UTF-8 -*-
"""
Created on 17.02.25

:author:     Martin Dočekal
"""
from pathlib import Path
from unittest import TestCase
from aicaller.template import StringTemplate, SegmentedStringTemplate, OpenAIMessageBuilder, OpenAITextContent, \
    OpenAIImageContent, OpenAIMultiModalMessageBuilder, OllamaMessageBuilder, MessagesTemplate

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
