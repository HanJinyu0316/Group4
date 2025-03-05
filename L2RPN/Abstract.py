from __future__ import annotations
import grid2op.Reward
import numpy as np
import grid2op
from abc import ABC, abstractmethod
from grid2op.Chronics import MultifolderWithCache
from grid2op.Environment import BaseEnv

class BaseEnvWrapper(ABC):

    def __init__(self, env_name, backend, grid2op_params):
        self.env:BaseEnv = grid2op.make(env_name, backend=backend, **grid2op_params)
        
        if "chronics_class" in grid2op_params.keys():
            if grid2op_params["chronics_class"] is MultifolderWithCache:
                self.env.chronics_handler.set_filter(lambda _: True)
                self.env.chronics_handler.reset()

    @abstractmethod
    def step(self, action:grid2op.Action.baseAction) -> tuple[np.ndarray, float, bool, dict]:
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def seed(self, seed:int) -> None:
        pass