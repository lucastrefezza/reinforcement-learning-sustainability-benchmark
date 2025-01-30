from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    actor = Model[0](envs).to(device)
    qf1 = Model[1](envs).to(device)
    qf2 = Model[1](envs).to(device)
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    # qf1 and qf2 are not used in this script
    qf1.load_state_dict(qf1_params)
    qf1.eval()
    qf2.load_state_dict(qf2_params)
    qf2.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            action, _, _ = actor.get_action(torch.Tensor(obs).to(device), deterministic=True)
        next_obs, _, _, _, infos = envs.step(action)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]

        obs = next_obs

    return episodic_returns
