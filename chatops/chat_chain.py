import os
import json
import time
import shutil
import logging
import pathlib
import importlib

from datetime import datetime

from camel.agents import RolePlaying
from camel.configs import ChatGPTConfig
from chatops.statistics import get_info
from camel.web_spider import modal_trans
from camel.typing import TaskType, ModelType
from chatops.utils import log_visualize, now
from chatops.chat_env import ChatEnv, ChatEnvConfig


class ChatChain:

    def __init__(
            self,
            config_path: str | os.PathLike[str] = None,
            config_phase_path: str | os.PathLike[str] = None,
            config_role_path: str | os.PathLike[str] = None,
            task_prompt: str = None,
            case_name: str = None,
            namespace: str = None,
            model_type: ModelType = ModelType.GPT_3_5_TURBO,
            docs_path: str | os.PathLike[str] = None,
            report_dir: str | os.PathLike[str] | None = None,
    ):
        # load config file
        self.config_path = pathlib.Path(config_path)
        self.config_phase_path = pathlib.Path(config_phase_path)
        self.config_role_path = pathlib.Path(config_role_path)
        self.case_name = case_name
        self.namespace = namespace
        self.model_type = model_type
        self.docs_path = pathlib.Path(docs_path)
        self.report_dir = pathlib.Path(report_dir) if report_dir else pathlib.Path(__file__).parent.parent / 'Report'

        self.config = json.loads(self.config_path.read_text(encoding='utf-8'))
        self.config_phase = json.loads(self.config_phase_path.read_text(encoding='utf-8'))
        self.config_role = json.loads(self.config_role_path.read_text(encoding='utf-8'))

        # init ChatChain config and recruitments
        self.chain = self.config['chain']
        self.recruitments = self.config['recruitments']
        self.web_spider = self.config['web_spider']

        # init default max chat turn
        self.chat_turn_limit_default = 10

        # init ChatEnv
        self.chat_env_config = ChatEnvConfig(
            clear_structure=self.config.get('clear_structure', True),
            visual_report=self.config.get('visual_report', False),
            incremental_mode=self.config.get('incremental_mode', False),
            background_prompt=self.config['background_prompt'],
        )
        self.chat_env = ChatEnv(self.chat_env_config)

        # the user input prompt will be self-improved (if set "self_improve": true in ChatChainConfig.json)
        # the self-improvement is done in self.preprocess
        self.task_prompt_raw = task_prompt

        # init role prompts
        self.role_prompts = {}
        for role in self.config_role:
            self.role_prompts[role] = '\n'.join(self.config_role[role])

        # init log
        self.start_time, self.log_path = self.get_log_path()

        # init SimplePhase instances
        # import all used phases in PhaseConfig.json from chatops.phase
        # note that in PhaseConfig.json there only exist SimplePhases
        # ComposedPhases are defined in ChatChainConfig.json and will be imported in self.execute_step
        self.compose_phase_module = importlib.import_module('chatops.composed_phase')
        self.phase_module = importlib.import_module('chatops.phase')
        self.phases = {}
        for phase in self.config_phase:
            assistant_role_name = self.config_phase[phase]['assistant_role_name']
            user_role_name = self.config_phase[phase]['user_role_name']
            phase_prompt = '\n\n'.join(self.config_phase[phase]['phase_prompt'])
            phase_class = getattr(self.phase_module, phase)
            phase_instance = phase_class(
                assistant_role_name=assistant_role_name,
                user_role_name=user_role_name,
                phase_prompt=phase_prompt,
                role_prompts=self.role_prompts,
                phase_name=phase,
                model_type=self.model_type,
                log_path=self.log_path,
                leader_role_name=self.config['leader_role_name'],
                consultant_role_name=self.config['consultant_role_name'],
            )
            self.phases[phase] = phase_instance

    def make_recruitment(self):
        """
        Recruit all employees
        """
        for employee in self.recruitments:
            self.chat_env.recruit(agent_name=employee)

    def execute_step(self, phase_item: dict):
        """
        Execute single phase in the chain
        Args:
            phase_item: single phase configuration in the ChatChainConfig.json
        """
        phase = phase_item['phase']
        phase_type = phase_item['phase_type']
        # For SimplePhase, just look it up from self.phases and conduct the "Phase.execute" method
        if phase_type == 'SimplePhase':
            max_turn_step = phase_item['max_turn_step']
            need_reflect = phase_item['need_reflect']
            if phase in self.phases:
                print("=======================")
                print(phase)
                print("=======================")
                self.chat_env = self.phases[phase].execute(
                    self.chat_env,
                    self.chat_turn_limit_default if max_turn_step <= 0 else max_turn_step,
                    need_reflect,
                )
            else:
                raise RuntimeError(f'Phase "{phase}" is not yet implemented in chatops.phase.')
        # For ComposedPhase, we create instance here then conduct the "ComposedPhase.execute" method
        elif phase_type == 'ComposedPhase':
            cycle_num = phase_item['cycle_num']
            composition = phase_item['composition']
            compose_phase_class = getattr(self.compose_phase_module, phase)
            if not compose_phase_class:
                raise RuntimeError(f'Phase "{phase}" is not yet implemented in chatops.compose_phase')
            compose_phase_instance = compose_phase_class(
                phase_name=phase,
                cycle_num=cycle_num,
                composition=composition,
                config_phase=self.config_phase,
                config_role=self.config_role,
                model_type=self.model_type,
                log_path=self.log_path,
                leader_role_name=self.config['leader_role_name'],
                consultant_role_name=self.config['consultant_role_name'],
            )
            self.chat_env = compose_phase_instance.execute(self.chat_env)
        else:
            raise RuntimeError(f'PhaseType "{phase_type}" is not yet implemented.')

    def execute_chain(self):
        """
        execute the whole chain based on ChatChainConfig.json
        Returns: None

        """
        for phase_item in self.chain:
            self.execute_step(phase_item)

    def get_log_path(self) -> tuple[str, pathlib.Path]:
        """
        get the log path (under the report path)
        Returns:
            start_time: time for starting the chatbot
            log_path: path to the log
        """
        start_time = now()
        case_name_full = '_'.join([self.case_name, self.namespace, start_time])
        directory = self.report_dir
        directory.mkdir(exist_ok=True, parents=True)
        log_path = directory / f'{case_name_full}.log'
        return start_time, log_path

    def pre_processing(self):
        """
        Remove useless files and log some global config settings
        """
        directory = self.report_dir

        if self.chat_env.config.clear_structure:
            for child in directory.iterdir():
                # logs with error trials are left in Report/
                if child.is_file() and child.suffix != '.log' and child.suffix != '.py':
                    try:
                        child.unlink()
                        print(f'{child} Removed.')
                    except OSError:
                        # NFS "Device or resource busy" — file still in use by another process;
                        # safe to skip, it will be cleaned up later
                        pass

        report_path = directory / '_'.join([self.case_name, self.namespace, self.start_time])
        self.chat_env.set_directory(report_path)

        # copy config files to report path
        shutil.copy(self.config_path, report_path)
        shutil.copy(self.config_phase_path, report_path)
        shutil.copy(self.config_role_path, report_path)

        # copy docs to report path in incremental mode
        if self.config['incremental_mode']:
            shutil.copytree(self.docs_path, report_path / 'base')
            self.chat_env.load_from_hardware(report_path / 'base')

        # write task prompt to report
        (report_path / f'{self.case_name}.prompt').write_text(self.task_prompt_raw, encoding='utf-8')

        preprocess_msg = '**[Preprocessing]**\n\n'
        llm_config = ChatGPTConfig()

        preprocess_msg += f'**ChatOps Starts** ({self.start_time})\n\n'
        preprocess_msg += f'**Timestamp**: {self.start_time}\n\n'
        preprocess_msg += f'**Config Path**: {self.config_path}\n\n'
        preprocess_msg += f'**config Phase Path**: {self.config_phase_path}\n\n'
        preprocess_msg += f'**Config Role Path**: {self.config_role_path}\n\n'
        preprocess_msg += f'**Task Prompt**: {self.task_prompt_raw}\n\n'
        preprocess_msg += f'**Case Name**: {self.case_name}\n\n'
        preprocess_msg += f'**Log File**: {self.log_path}\n\n'
        preprocess_msg += f'**ChatOpsConfig**:\n{self.chat_env.config.__str__()}\n\n'
        preprocess_msg += f'**LLMConfig**:\n{llm_config}\n\n'
        log_visualize(preprocess_msg)

        # init task prompt
        if self.config['self_improve']:
            # TODO: Make self-improvement usable
            raise NotImplementedError
            self.chat_env.env_dict['task_prompt'] = self.self_task_improve(self.task_prompt_raw)
        else:
            self.chat_env.env_dict['task_prompt'] = self.task_prompt_raw
        if self.web_spider:
            # TODO: Make web-spider usable
            raise NotImplementedError
            self.chat_env.env_dict['task_description'] = modal_trans(self.task_prompt_raw)

    def post_processing(self):
        """
        Summarize the production and move log files to the report directory
        """
        self.chat_env.write_meta()

        post_info = '**[Post Info]**\n\n'
        now_time = now()
        time_format = '%Y%m%d%H%M%S'
        datetime1 = datetime.strptime(self.start_time, time_format)
        datetime2 = datetime.strptime(now_time, time_format)
        duration = (datetime2 - datetime1).total_seconds()

        post_info += 'Report Info: {}'.format(
            get_info(
                self.chat_env.env_dict['directory'],
                self.log_path,
                self.model_type,
            ) +
            f'\n\n🕑**duration**={duration:.2f}s\n\n'
        )
        post_info += f'ChatOps Starts ({self.start_time})\n\n'
        post_info += f'ChatOps Ends ({now_time})\n\n'

        log_visualize(post_info)

        logging.shutdown()
        time.sleep(1)

        case_name_full = '_'.join([self.case_name, self.namespace, self.start_time])
        report_subdir = self.report_dir / case_name_full
        report_subdir.mkdir(parents=True, exist_ok=True)
        shutil.move(
            self.log_path,
            report_subdir / f'{case_name_full}.log'
        )

    # @staticmethod
    def self_task_improve(self, task_prompt: str) -> str:
        """
        Ask agent to improve the user query prompt
        Args:
            task_prompt: original user query prompt
        Returns:
            revised_task_prompt: revised prompt from the prompt engineer agent
        """
        self_task_improve_prompt = """I will give you a short description of a software design requirement, 
please rewrite it into a detailed prompt that can make large language model know how to make this software better based this prompt,
the prompt should ensure LLMs build a software that can be run correctly, which is the most import part you need to consider.
remember that the revised prompt should not contain more than 200 words, 
here is the short description:\"{}\". 
If the revised prompt is revised_version_of_the_description, 
then you should return a message in a format like \"<INFO> revised_version_of_the_description\", do not return messages in other formats.""".format(
            task_prompt)
        role_play_session = RolePlaying(
            assistant_role_name="Prompt Engineer",
            assistant_role_prompt="You are an professional prompt engineer that can improve user input prompt to make LLM better understand these prompts.",
            user_role_prompt="You are an user that want to use LLM to build software.",
            user_role_name="User",
            task_type=TaskType.CHATOPS,
            task_prompt="Do prompt engineering on user query",
            with_task_specify=False,
            model_type=self.model_type,
        )

        # log_visualize("System", role_play_session.assistant_sys_msg)
        # log_visualize("System", role_play_session.user_sys_msg)

        _, input_user_msg = role_play_session.init_chat(None, None, self_task_improve_prompt)
        assistant_response, user_response = role_play_session.step(input_user_msg, True)
        revised_task_prompt = assistant_response.msg.content.split("<INFO>")[-1].lower().strip()
        log_visualize(role_play_session.assistant_agent.role_name, assistant_response.msg.content)
        log_visualize(
            "**[Task Prompt Self Improvement]**\n**Original Task Prompt**: {}\n**Improved Task Prompt**: {}".format(
                task_prompt, revised_task_prompt))
        return revised_task_prompt
