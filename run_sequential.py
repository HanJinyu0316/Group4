import warnings, math, copy, logging
import numpy as np
import torch
import pickle
import ray
from optuna.trial import TrialState
from jsonargparse import auto_cli
from pathlib import Path
from grid2op.Reward import BaseReward
from L2RPN.Abstract import Stage, SequentialConfig
from L2RPN.Agents.Exhaustive import ExhaustiveAgent
from L2RPN.Agents.DoNothing import DoNothingAgent
from L2RPN.Visualization.Comparison import bar_comparison
from L2RPN.Training.Common import setup_optuna_study, setup_exp_dir
from L2RPN.Training.OffPolicySequential import SequentialTrainer, sequential_trial_objective
    
def main(name:str, path:str="", debug:bool=False,  suffix:str="", interactive:bool=False,
         replay:bool=False, benchmark:bool=False, replay_stage:Stage=Stage.VALIDATE):
    """
    Sequential Process, a single worker both collects experiences and updates its policy. Consequently this process
    is completely deterministic. Different Hyperparameter options can be explored via Optuna, with support for
    parallelized trials.

    Args:
        name (str, optional): Name of sequential config to use. Defaults to "".
        path (str, optional): Path of Config file to use, takes priority over config name.. Defaults to "".
        debug (bool, optional): Whether to run the Sequential workflow in Debug mode. Defaults to False.
        suffix (str, optional): Suffix to add to Optuna study name. Defaults to "".
        interactive (bool, optional): Whether to run the script interactively.. Defaults to False.
        replay (bool, optional): Replay best agent from Optuna study. Defaults to False.
        benchmark (bool, optional): Replay best Trained agent alongside Do Nothing and Exhaustive agents. Defaults to False.
        replay_stage (Stage, optional): Which stage to replay. Defaults to Stage.VALIDATE.

    Raises:
        KeyError: No Sequential Config class is provided
        EOFError: _description_
    """
    config = SequentialConfig(Path(__file__).parent / "Configs" / f"{name}.yaml")
    config.load_config()
    
    if torch.cuda.is_available():
        print(f"GPU Usage Active!\nNo. of GPUs: {torch.cuda.device_count()}, Currently Active: gpu:{torch.cuda.current_device()}")
    else:
        print(f"CPU Usage Active!\nNo. of CPUs: {torch.cpu.device_count()}, Currently Active: cpu:{torch.cpu.current_device()}")
    
    # >> Train single agent (no hyperparameter optimization) <<
    config = SequentialConfig.convert_hparams(config, optuna=False)
    trainer = SequentialTrainer(config)

    latest_ep_no, total_activations, final_activations, epoch_no = 0, 0, 0, 1
    # Run 1 validation loop 1st (acts as baseline)
    if replay:
        trainer = SequentialConfig(config)
        best_val_reward, _, _, _ = trainer.loop(Stage.VALIDATE)
    else:
        best_val_reward, _, _, _ = trainer.loop(Stage.VALIDATE)
        trainer.save_checkpoint(suffix="Best")

        while latest_ep_no < trainer.max_iter:
            avg_train_reward, latest_ep_no, activations, _ = trainer.loop(Stage.TRAIN, start_ep_no=latest_ep_no, 
                                                                            final_activations=final_activations)
            total_activations += activations
            avg_val_reward, _, _, _ = trainer.loop(Stage.VALIDATE)
            if avg_val_reward > best_val_reward:
                best_val_reward = avg_val_reward
                trainer.save_checkpoint(suffix="Best")
            trainer.log(f"Epoch {epoch_no} :: Avg. Train Reward {avg_train_reward} :: Avg. Validation Reward {avg_val_reward}")
            trainer.save_checkpoint(suffix="Latest")

            # Final Stage: Continue X active steps after reaching minimum epsilon
            if math.isclose(trainer.epsilon, trainer.min_eps):
                trainer.log(f"Final Stage {final_activations} > {config.trainer_config.n_activations_after_min_eps}?")
                final_activations += activations
                if final_activations > config.trainer_config.n_activations_after_min_eps:
                    trainer.log(f"Stopping since {final_activations} active training steps have occured since reaching minimum epsilon.")
                    break
            epoch_no += 1

if __name__ == "__main__":
    auto_cli(main)

    