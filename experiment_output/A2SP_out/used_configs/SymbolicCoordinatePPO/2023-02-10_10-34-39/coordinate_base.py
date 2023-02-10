### THIS FILE ORIGINALLY LOCATED AT '/home/zidong/work/allenact/projects/ithor_A2SP/configs/coordinate_base.py'

import copy
import platform
from abc import abstractmethod
from typing import Optional, List, Sequence, Dict, Any, Tuple

import ai2thor.platform
import gym.spaces
import stringcase
import torch
import torchvision.models
from torch import nn, cuda, optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)

from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, Sensor, ExpertActionSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import TrainingPipeline, LinearDecay, Builder
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays


import data_generator.gen_data_utils as gen_data_utils

from A2SP.multi_agent_env import MultiAgentEvneronment
from A2SP.constants import (
    AgentType,
    Mode,
    THOR_COMMIT_ID, 
    ROTATE_DEGREES,
    VISIBILITY_DISTANCE,
    SCREEN_SIZE,
    ALL_ACTIONS_NAME,
    USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS,
    PICKUP_ACTIONS,
    PUT_ACTIONS,
    OPEN_ACTIONS
)
from A2SP.models import CoordinateActorCristicRNN
from A2SP.tasks import CoordinateTaskSampler

class CoordinateBaseExperimentConfig(ExperimentConfig):
    # Task parameters
    MAX_STEPS = 200
    REQUIRE_DONE_ACTION = False

    # Environment parameters
    MODE = Mode.SYMBOLIC
    AGENTS_NUM = 2
    MA_ENV_KWARGS = {
        'agents_type': [AgentType.AGENT_WITH_FULL_CAPABILITIES, AgentType.AGENT_WITH_FULL_CAPABILITIES],
        'main_agent_id': 0,
        'agents_num': AGENTS_NUM,
        'mode': MODE,
    }

    # controller parameters
    THOR_CONTROLLER_KWARGS = dict(
        # controller setting
        commit_id = THOR_COMMIT_ID,
        fastActionEmit = True,
        # agent configuration 
        agentCount = AGENTS_NUM,
        agentMode="default",
        visibilityDistance=VISIBILITY_DISTANCE,
        # Navigation properties
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=ROTATE_DEGREES,
        # camera properties
        width=SCREEN_SIZE,
        height=SCREEN_SIZE,
        fieldOfView=90,
        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )

    # Model parameters
    AGENTS_EMBED_LENGTH = 64
    HIDDEN_SIZE = 512
    RNN_LAYERS_NUM = 1
    RNN_TYPE = 'GRU'


    # Training parameters
    TRAINING_STEPS = int(1e5)   # 75e6
    SAVE_INTERVAL = int(1e4)    # 1e6

    # Sensor info
    SENSORS: Optional[Sequence[Sensor]] = None

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return cls.SENSORS

    @classmethod
    def actions(cls):
        function_actions = ['Wait', 'Done'] if cls.REQUIRE_DONE_ACTION else ['Wait'] 
        return (
            *ALL_ACTIONS_NAME['navigation_actions'],
            *PICKUP_ACTIONS,
            *PUT_ACTIONS,
            *OPEN_ACTIONS,
            *function_actions,
        )

    @classmethod
    def get_lr_scheduler_builder(cls, use_lr_decay: bool):
        return (
            None
            if not use_lr_decay
            else Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(
                        steps=cls.TRAINING_STEPS // 3, startp=1.0, endp=1.0 / 3
                    )
                },
            )
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> MachineParams:
        '''
        Return the number of processes and gpu_ids to use with training.
        '''
        num_gpus = cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        if mode == "train":
            nprocesses = cls.num_train_processes() if torch.cuda.is_available() else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )
        elif mode == "valid":
            devices = [num_gpus - 1] if has_gpu else [torch.device("cpu")]
            nprocesses = 2 if has_gpu else 0
        else:
            nprocesses = 2 if has_gpu else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )

        nprocesses = split_processes_onto_devices(
            nprocesses=nprocesses, ndevices=len(devices)
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices,
        )
 
    @classmethod
    def stagewise_task_sampler_args(
        cls,
        stage: str,
        process_ind: int,
        total_processes: int,
        allowed_ids_subset: Optional[Sequence[int]] = None,
        allowed_scenes: Sequence[str] = None,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        if allowed_scenes is not None:
            scenes = allowed_scenes
        elif stage == "combined":
            # Split scenes more evenly as the train scenes will have more episodes
            train_scenes = gen_data_utils.get_scenes_name("train")
            other_scenes = gen_data_utils.get_scenes_name("val") + gen_data_utils.get_scenes_name(
                "test"
            )
            assert len(train_scenes) == 2 * len(other_scenes)
            scenes = []
            while len(train_scenes) != 0:
                scenes.append(train_scenes.pop())
                scenes.append(train_scenes.pop())
                scenes.append(other_scenes.pop())
            assert len(train_scenes) == len(other_scenes)
        else:
            scenes = gen_data_utils.get_scenes_name(stage)

        if total_processes > len(scenes):
            assert stage == "train" and total_processes % len(scenes) == 0
            scenes = scenes * (total_processes // len(scenes))

        allowed_scenes = list(
            sorted(partition_sequence(seq=scenes, parts=total_processes,)[process_ind])
        )

        scene_to_allowed_ids = None
        if allowed_ids_subset is not None:
            allowed_ids_subset = tuple(allowed_ids_subset)
            assert stage in ["valid", "train_unseen"]
            scene_to_allowed_ids = {
                scene: allowed_ids_subset for scene in allowed_scenes
            }
        seed = md5_hash_str_as_int(str(allowed_scenes))

        device = (
            devices[process_ind % len(devices)]
            if devices is not None and len(devices) > 0
            else torch.device("cpu")
        )
        x_display: Optional[str] = None
        gpu_device: Optional[int] = None
        thor_platform: Optional[ai2thor.platform.BaseLinuxPlatform] = None
        if platform.system() == "Linux":
            try:
                x_displays = get_open_x_displays(throw_error_if_empty=True)
             
                if devices is not None and len(
                    [d for d in devices if d != torch.device("cpu")]
                ) > len(x_displays):
                    get_logger().warning(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                x_display = x_displays[process_ind % len(x_displays)]
            except IOError: # Actually, CloudRendering is used on severs. 
                # Could not find an open `x_display`, use CloudRendering instead.
                assert all(
                    [d != torch.device("cpu") and d >= 0 for d in devices]
                ), "Cannot use CPU devices when there are no open x-displays as CloudRendering requires specifying a GPU."
                gpu_device = device
                thor_platform = ai2thor.platform.CloudRendering

        kwargs = {
            "stage": stage,
            "allowed_scenes": allowed_scenes,
            "scene_to_allowed_ids": scene_to_allowed_ids,
            "seed": seed,
            "x_display": x_display,
            "thor_controller_kwargs": {
                "gpu_device": gpu_device,
                "platform": thor_platform,
            },
        }
        
        sensors = kwargs.get("sensors", copy.deepcopy(cls.sensors()))
        kwargs["sensors"] = sensors

        return kwargs

    @classmethod
    def train_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        devices = [0, 1]
        return dict(
            force_cache_reset=False,
            epochs=float("inf"),
            **cls.stagewise_task_sampler_args(
                stage="train",
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    def valid_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=True,
            epochs=1,
            **cls.stagewise_task_sampler_args(
                stage="valid",
                allowed_rearrange_inds_subset=tuple(range(0, 50, 5)),
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    def test_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        task_spec_in_metrics: bool = False,
    ):
        task_spec_in_metrics = False

        # Train_unseen
        # stage = "train_unseen"
        # allowed_rearrange_inds_subset = list(range(15))

        # Val
        # stage = "val"
        # allowed_rearrange_inds_subset = None

        # Test
        # stage = "test"
        # allowed_rearrange_inds_subset = None

        # Combined (Will run inference on all datasets)
        stage = "combined"
        allowed_rearrange_inds_subset = None

        return dict(
            force_cache_reset=True,
            epochs=1,
            task_spec_in_metrics=task_spec_in_metrics,
            **cls.stagewise_task_sampler_args(
                stage=stage,
                allowed_rearrange_inds_subset=allowed_rearrange_inds_subset,
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: int,
        scene_to_allowed_ids: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> CoordinateTaskSampler:
        '''
        Return a CoordinateTaskSampler.
        '''
        sensors = cls.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        
        return CoordinateTaskSampler.from_fixed_dataset(
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_ids=scene_to_allowed_ids,
            ma_env_kwargs=dict(
                **cls.MA_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **cls.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            epochs=epochs,
            **kwargs,
        )

    @classmethod
    @abstractmethod
    def _training_pipeline_info(cls) -> Dict[str, Any]: # This function is defined in the subclasses
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def num_train_processes(cls) -> int:    # This function is defined in the subclasses
        raise NotImplementedError

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        info = cls._training_pipeline_info()

        return TrainingPipeline(
            gamma=info.get("gamma", 0.99),
            use_gae=info.get("use_gae", True),
            gae_lambda=info.get("gae_lambda", 0.95),
            num_steps=info["num_steps"],
            num_mini_batch=info["num_mini_batch"],
            update_repeats=info["update_repeats"],
            max_grad_norm=info.get("max_grad_norm", 0.5),
            save_interval=cls.SAVE_INTERVAL,
            named_losses=info["named_losses"],
            metric_accumulate_interval=(
                cls.num_train_processes() * cls.MAX_STEPS
                if torch.cuda.is_available()
                else 1
            ),
            optimizer_builder=Builder(optim.Adam, dict(lr=info["lr"])),
            advance_scene_rollout_period=None,
            pipeline_stages=info["pipeline_stages"],
            lr_scheduler_builder=cls.get_lr_scheduler_builder(
                use_lr_decay=info["use_lr_decay"]
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return CoordinateActorCristicRNN(
            action_space=gym.spaces.Tuple(
                [
                    gym.spaces.Discrete(len(cls.actions()))
                    for _ in range(cls.AGENTS_NUM)
                ]
            ), 
            # action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=SensorSuite(cls.sensors()).observation_spaces,
            agents_num=cls.AGENTS_NUM,
            agents_embed_length=cls.AGENTS_EMBED_LENGTH,
            hidden_size=cls.HIDDEN_SIZE,
            rnn_layers_num=cls.RNN_LAYERS_NUM,
            rnn_type=cls.RNN_TYPE
        )