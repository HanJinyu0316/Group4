import json, time
import numpy as np
from typing import Tuple, Type
from pathlib import Path

import torch
from grid2op.Action import BaseAction
from grid2op.Backend import Backend
from grid2op.Episode import CompactEpisodeData
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import RESET_OPTIONS_TYPING
from grid2op.Reward import BaseReward, GameplayReward

from L2RPN.Abstract import BaseEnvWrapper
from L2RPN.Utilities.Progress import get_env_size

from .ActionPreProcessing import (ActionPipelineParams, ActionPipeline, Support, ActionFilters)
from .OutcomePostProcessing import (OutcomePipelineParams, OutcomePipeline, OutcomeFilters)
from .Utilities import VanillaTracker
from ..Abstract import Stage

class TemplateEnvWrapper(BaseEnvWrapper):
    
    def __init__(
        self,
        env_name:str,
        backend:Backend,
        env_kwargs:dict,
        *args,
        rho_threshold:float=0.95,
        verbose:bool=False,
        **kwargs
    ):
        if not isinstance(backend, Backend):
            try:
                backend = backend()
            except TypeError:
                print("A backend instance was not provided and initialisation failed. Please provide a valid backend or backend class")
                return
        super().__init__(env_name=env_name, backend=backend, grid2op_params=env_kwargs)
        # >> Simple Attributes <<
        # Used for tracking variables across an episode / step
        self.rho_threshold = rho_threshold # Fraction of thermal limit beyomnd which agent is activated
        self.tracker:VanillaTracker = VanillaTracker() # Utility to help keep track of reward, observation, etc.
        self.verbose = verbose

    def get_env_size(self) -> tuple[int, list[str]]:
        return get_env_size(self.env)
    
    def process_agent_action(self, action) -> BaseAction:
        """
        Process the action suggested by the agent.

        Args:
            action (object): Agent's output (the action)
        """
        pass

    def convert_observation(self, observation:BaseObservation) -> np.ndarray:
        """
        Convert a Grid2Op observation of the environment's state into a form the
        agent can understand.

        Args:
            observation (BaseObservation): Observation of grid's state

        Returns:
            numpy.ndarray: Numpy array that is the input to the agent (will be converted to torch inside the agent)
        """
        pass
    
    def step(self, agent_action) -> tuple[np.ndarray, float, bool, dict]:
        """
        Provides an interface to interact with the wrapped Grid2Op environment.

        Args:
            action (int): integer representation of the action

        Returns:
            tuple[np.ndarray, float, bool, dict]: 
            Observation: vector or graph representation of the environment state
            reward: total accumulated reward over the non active timesteps
            done: if the scenario finished
            info: regular grid2op info. additionally provides a mask for illegal actions, as well as the number of steps taken in the grid2op environment
        """        
        self.tracker.reset_step()

        action = self.process_agent_action(agent_action)
        
        # >> Execute Agent's Action <<
        obs, reward, done, info = self.env.step(action)
        self.tracker.step(obs, reward, done, info)
        
        # >> Step While Safe <<
        # Step through environment so long as line loading is under threshold
        self._step_while_safe()

        self.tracker.info.update({"time":time.perf_counter() - self.tracker.start})
        obs_vec = self.convert_observation(self.tracker.state)
        return (obs_vec, # Vector Representation of the Observation
                self.tracker.tot_reward, # Reward accumulated, can be a sum if we include heuristics, otherwise is just the reward from env.step(...)
                self.tracker.done, # Whether the episode ended, occurs if we reach the end of the timeseries, or the powerflow diverges
                self.tracker.info # Additional information, stored in a dictionary
        ) 
        
    def _step_while_safe(self):
        """
        Keep stepping through environment until agent is activated again (or episode ends)
        """
        while not self.tracker.done and not np.any(self.tracker.state.rho >= self.rho_threshold):
            action_ = self.env.action_space({}) # Do Nothing
            obs, reward, done, info = self.env.step(action_)
            self.tracker.step(obs, reward, done, info)


    def reset(self, seed:int|None=None, options:RESET_OPTIONS_TYPING={}) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Reset the environment, this will start a new episode
        
        Returns:
            np.ndarray | Data | Any: observation, type depends
                on conversion routine.
            dict[str]: Information, contains the following:
                "action_mask": None
                "steps_taken": int
                "reward": 0.0
            bool: Whether the episode has been terminated/truncated,
                should be False.
        """
        
        # >> Episode ID <<
        if "time serie id" in options:
            ep_id = options["time serie id"]
        else:
            ep_id = self.env.chronics_handler.get_name()
            options["time serie id"] = ep_id
        
        # >> Reset Environment to Target Episode <<
        # NOTE: Options can overwrite the init_ts
        self.tracker.reset_episode(
            self.env.reset(options=options)
        )
        
        obs_vec = self.convert_observation(self.tracker.state)
        return (obs_vec, # Obs
                dict(reward=0), # Info
                self.tracker.done # Done
        ) 

    def set_id(self, chronic_id:int|str):
        self.env.set_id(chronic_id)
    
    def seed(self, seed:int) -> None:
        self.env.seed(seed=seed)

    def max_episode_duration(self) -> int:
        return self.env.max_episode_duration()
            
        
