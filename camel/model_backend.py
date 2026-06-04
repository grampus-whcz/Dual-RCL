# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import importlib

from abc import ABC, abstractmethod
from typing import Any, Dict

import tiktoken

from camel.typing import ModelType
from chatops.statistics import prompt_cost
from chatops.utils import log_visualize
from camel.utils import get_model_token_limit

import os


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, Qianfan API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from LLM API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict
        self.api_key = os.environ.get('OPENAI_API_KEY', 'api_key')
        self.base_url = os.environ.get('BASE_URL', 'http://localhost:8000/v1')
        self.openai = importlib.import_module('openai')

    def run(self, *args, **kwargs):
        string = "\n".join([message["content"] for message in kwargs["messages"]])
        #encoding = tiktoken.encoding_for_model(self.model_type.value)
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-1106')
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        if self.base_url:
            client = self.openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            client = self.openai.OpenAI(
                api_key=self.api_key,
            )

            num_max_token = get_model_token_limit(self.model_type)
            num_max_completion_tokens = num_max_token - num_prompt_tokens
            self.model_config_dict['max_tokens'] = num_max_completion_tokens

        response = client.chat.completions.create(*args, **kwargs, model=self.model_type.value,
                                                  **self.model_config_dict)
        # For Ollama models with reasoning (e.g. qwen3), the thinking tokens
        # count toward max_tokens.  If content comes back empty because the
        # model used all tokens for reasoning, retry with a larger budget.
        if (not response.choices[0].message.content
                and response.choices[0].finish_reason == 'length'):
            self.model_config_dict['max_tokens'] = min(
                self.model_config_dict.get('max_tokens', 2048) * 4,
                16384,
            )
            response = client.chat.completions.create(
                *args, **kwargs, model=self.model_type.value,
                **self.model_config_dict)

        cost = prompt_cost(
            self.model_type.value,
            num_prompt_tokens=response.usage.prompt_tokens,
            num_completion_tokens=response.usage.completion_tokens
        )

        log_visualize(
            "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                response.usage.prompt_tokens, response.usage.completion_tokens,
                response.usage.total_tokens, cost))
        from openai.types.chat import ChatCompletion
        if not isinstance(response, ChatCompletion):
            raise RuntimeError("Unexpected return from OpenAI API")
        return response


class QianfanModel(ModelBackend):
    r"""Qianfan API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

        import qianfan
        self.client = qianfan.ChatCompletion()

    def run(self, *args, **kwargs):
        response = self.client.do(
            model=self.model_type.value,
            messages=kwargs["messages"],
        ).body
        cost = prompt_cost(
            self.model_type.value,
            num_prompt_tokens=response['usage']['prompt_tokens'],
            num_completion_tokens=response['usage']['completion_tokens'],
        )
        log_visualize(
            "**[Qianfan_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                response['usage']['prompt_tokens'], response['usage']['completion_tokens'],
                response['usage']['total_tokens'], cost))
        print(response)
        return response


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.GPT_3_5_TURBO

        if model_type in {
            ModelType.GPT_3_5_TURBO,
            ModelType.GPT_4,
            ModelType.GPT_4_32K,
            ModelType.GPT_4_TURBO,
            ModelType.GPT_4_TURBO_V,
            ModelType.MISTRAL_7B,
            ModelType.LLAMA_3_8B,
            ModelType.DEEPSEEK_R1_0528,
            ModelType.OLLAMA_QWEN3_14B,
            ModelType.OLLAMA_QWEN3_8B,
            None
        }:
            model_class = OpenAIModel
        elif model_type in {
            ModelType.ERNIE_BOT_4,
        }:
            model_class = QianfanModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        # log_visualize("Model Type: {}".format(model_type))
        inst = model_class(model_type, model_config_dict)
        return inst
