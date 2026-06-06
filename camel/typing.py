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
from enum import Enum


class TaskType(Enum):
    AI_SOCIETY = "ai_society"
    CODE = "code"
    MISALIGNMENT = "misalignment"
    TRANSLATION = "translation"
    EVALUATION = "evaluation"
    SOLUTION_EXTRACTION = "solution_extraction"
    CHATOPS = "chat_dev"
    DEFAULT = "default"


class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    CRITIC = "critic"
    EMBODIMENT = "embodiment"
    DEFAULT = "default"
    CHATDEV = "AgentTech"
    CHATDEV_CONSULTANT = "consultant"
    CHATDEV_CEO = "chief executive officer (CEO)"
    CHATDEV_CHRO = "chief human resource officer (CHRO)"
    CHATDEV_CPO = "chief product officer (CPO)"
    CHATDEV_CTO = "chief technology officer (CTO)"
    CHATDEV_PROGRAMMER = "programmer"
    CHATDEV_REVIEWER = "code reviewer"
    CHATDEV_TESTER = "software test engineer"
    CHATDEV_CCO = "chief creative officer (CCO)"


class ModelType(Enum):
    # OpenAI
    GPT_3_5_TURBO = 'gpt-3.5-turbo-1106'
    GPT_4 = 'gpt-4'
    GPT_4_32K = 'gpt-4-32k'
    GPT_4_TURBO = 'gpt-4-1106-preview'
    GPT_4_TURBO_V = 'gpt-4-1106-vision-preview'
    # Qianfan
    ERNIE_BOT_4 = 'ERNIE-Bot-4'
    # stub
    STUB = 'stub'
    # 新增加的
    MISTRAL_7B = 'mistralai/mistral-7b-instruct-v0.3'
    LLAMA_3_8B = 'Meta-Llama-3.1-8B-Instruct'#'meta/llama-3.1-8b-instruct'
    DEEPSEEK_R1_0528 = 'deepseek-r1-0528'
    # Ollama local models
    OLLAMA_QWEN3_14B = 'qwen3:14b'
    OLLAMA_QWEN3_8B = 'qwen3:8b'
    # ZhipuAI GLM models (Coding endpoint)
    GLM_4_5 = 'glm-4.5'
    GLM_4_7 = 'glm-4.7'

    @property
    def value_for_tiktoken(self):
        return self.value if self.name not in ('MISTRAL_7B', 'LLAMA_3_8B', 'DEEPSEEK_R1_0528',
                                                'OLLAMA_QWEN3_14B', 'OLLAMA_QWEN3_8B',
                                                'GLM_4_5', 'GLM_4_7') else 'gpt-3.5-turbo-1106'


class ModelProviderType(Enum):
    OPENAI = 'OpenAI'
    QIANFAN = 'Qianfan'
    STUB = 'stub'


class PhaseType(Enum):
    REFLECTION = "reflection"
    RECRUITING_CHRO = "recruiting CHRO"
    RECRUITING_CPO = "recruiting CPO"
    RECRUITING_CTO = "recruiting CTO"
    DEMAND_ANALYSIS = "demand analysis"
    CHOOSING_LANGUAGE = "choosing language"
    RECRUITING_PROGRAMMER = "recruiting programmer"
    RECRUITING_REVIEWER = "recruiting reviewer"
    RECRUITING_TESTER = "recruiting software test engineer"
    RECRUITING_CCO = "recruiting chief creative officer"
    CODING = "coding"
    CODING_COMPLETION = "coding completion"
    CODING_AUTOMODE = "coding auto mode"
    REVIEWING_COMMENT = "review comment"
    REVIEWING_MODIFICATION = "code modification after reviewing"
    ERROR_SUMMARY = "error summary"
    MODIFICATION = "code modification"
    ART_ELEMENT_ABSTRACTION = "art element abstraction"
    ART_ELEMENT_INTEGRATION = "art element integration"
    CREATING_ENVIRONMENT_DOCUMENT = "environment document"
    CREATING_USER_MANUAL = "user manual"


__all__ = ["TaskType", "RoleType", "ModelType", "PhaseType"]
