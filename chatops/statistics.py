import os
import pathlib
import numpy as np

from camel.typing import ModelType


def prompt_cost(model_type: ModelType, num_prompt_tokens: int, num_completion_tokens: int) -> float:
    input_cost_map = {
        ModelType.GPT_3_5_TURBO: 0.001,
        ModelType.GPT_4: 0.03,
        ModelType.GPT_4_32K: 0.06,
        ModelType.GPT_4_TURBO: 0.01,
        ModelType.GPT_4_TURBO_V: 0.01,
        ModelType.ERNIE_BOT_4: 0.02,
    }
    output_cost_map = {
        ModelType.GPT_3_5_TURBO: 0.002,
        ModelType.GPT_4: 0.06,
        ModelType.GPT_4_32K: 0.12,
        ModelType.GPT_4_TURBO: 0.03,
        ModelType.GPT_4_TURBO_V: 0.03,
        ModelType.ERNIE_BOT_4: 0.02,
    }
    if model_type not in input_cost_map or model_type not in output_cost_map:
        return 0
    return (num_prompt_tokens * input_cost_map[model_type] / 1000.0 +
            num_completion_tokens * output_cost_map[model_type] / 1000.0)


def get_info(report_path: str | os.PathLike[str], log_path: str | os.PathLike[str], model_type: ModelType) -> str:
    report_path = pathlib.Path(report_path)
    log_path = pathlib.Path(log_path)
    print('Report Path:', report_path)

    num_png_files = -1
    num_doc_files = -1
    num_utterance = -1
    num_reflection = -1
    num_prompt_tokens = -1
    num_completion_tokens = -1
    num_total_tokens = -1

    if report_path.exists():
        files = report_path.iterdir()
        num_doc_files = len([filename for filename in files if filename.suffix == '.md'])
        num_png_files = len([filename for filename in files if filename.suffix == '.png'])

        lines = log_path.read_text(encoding='utf-8').splitlines()
        start_lines = [line for line in lines if '**[Start Chat]**' in line]
        chat_lines = [line for line in lines if '<->' in line]
        num_utterance = len(start_lines) + len(chat_lines)

        sublines = [line for line in lines if line.startswith('prompt_tokens:')]
        if len(sublines) > 0:
            nums = [int(line.split(': ')[-1]) for line in sublines]
            num_prompt_tokens = int(np.sum(nums))

        sublines = [line for line in lines if line.startswith('completion_tokens:')]
        if len(sublines) > 0:
            nums = [int(line.split(': ')[-1]) for line in sublines]
            num_completion_tokens = int(np.sum(nums))

        sublines = [line for line in lines if line.startswith('total_tokens:')]
        if len(sublines) > 0:
            nums = [int(line.split(': ')[-1]) for line in sublines]
            num_total_tokens = int(np.sum(nums))

        num_reflection = 0
        for line in lines:
            if 'on : Reflection' in line:
                num_reflection += 1

    cost = 0.0
    if num_png_files != -1:
        cost += num_png_files * 0.016
    cost += prompt_cost(model_type, num_prompt_tokens, num_completion_tokens)

    info = (
        f'\n\n💰**cost**=${cost:.6f}'
        f'\n\n🏞**num_png_files**={num_png_files}'
        f'\n\n📚**num_doc_files**={num_doc_files}'
        f'\n\n🗣**num_utterances**={num_utterance}'
        f'\n\n🤔**num_self_reflections**={num_reflection}'
        f'\n\n❓**num_prompt_tokens**={num_prompt_tokens}'
        f'\n\n❗**num_completion_tokens**={num_completion_tokens}'
        f'\n\n🌟**num_total_tokens**={num_total_tokens}'
    )
    return info
