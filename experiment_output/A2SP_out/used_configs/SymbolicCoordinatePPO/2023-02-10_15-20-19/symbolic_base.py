### THIS FILE ORIGINALLY LOCATED AT '/home/zidong/work/allenact/projects/ithor_A2SP/configs/symbolic_representation/symbolic_base.py'

from abc import ABC
from typing import Optional, Dict, Sequence

from allenact.base_abstractions.sensor import SensorSuite, Sensor


from configs.coordinate_base import CoordinateBaseExperimentConfig
from A2SP.sensors import (
    SymbolicObjectSensor,
    SymbolicAgentSensor
)
from A2SP.constants import Mode

class SymbolicCoordinateExperimentConfig(CoordinateBaseExperimentConfig, ABC):
    MODE = Mode.SYMBOLIC

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [
            SymbolicObjectSensor(),
            SymbolicAgentSensor()
        ]



