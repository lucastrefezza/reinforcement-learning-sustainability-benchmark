import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from codecarbon import EmissionsTracker


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rlsb"
    """the wandb's project name"""
    wandb_entity: str = "rlsb"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4 # ppo default: 2.5e-4, but uses annealing
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    gamma: float = 0.99
    """the discount factor gamma"""
    # keep it for now
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # env = ClipRewardEnv(env) used in DQN to stabilize the updates,
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7 , 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        )

    def forward(self, x):
        logits = self.network(x / 255.0)
        return Categorical(logits=logits)


def compute_returns(rewards, gamma=0.99):
    returns = []
    gain = 0 # because "return" is obviously restricted

    for reward in reversed(rewards):
        gain = reward + gamma * gain
        returns.insert(0, gain)

    returns = torch.tensor(returns, dtype=torch.float32)
    # normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


if __name__ == "__main__":
    args = tyro.cli(Args)
    date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(int(time.time())))
    run_name = f"{args.exp_name}__{args.env_id}__{args.seed}__{date_time}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    policy = PolicyNetwork(envs).to(device)
    # eps for stability since we use gradients from a stochastic policy, value from literature
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)


    # Code Carbon tracking
    # tracker = EmissionsTracker(
    #     project_name="rlsb",
    #     output_dir="emissions",
    #     experiment_id=run_name,
    #     experiment_name=run_name,
    #     tracking_mode="process",
    #     log_level="warning",
    #     on_csv_write="append",
    # )
    # tracker.start()

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    global_step = 0
    done = False
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.Tensor(obs).to(device)

    while global_step < args.total_timesteps:
        # storage setup
        states = []
        actions = []
        log_probs = []
        rewards = []

        # generate an episode
        while not done:
            actions_distribution = policy(obs)
            action = actions_distribution.sample()
            log_prob = actions_distribution.log_prob(action)

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = terminations[0] or truncations[0]
            states.append(obs)
            actions.append(action)
            rewards.append(reward[0])
            log_probs.append(log_prob)
            obs = torch.Tensor(next_obs).to(device)
            global_step += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # compute returns
        returns = compute_returns(rewards, args.gamma).to(device)

        # update the policy
        loss = -torch.sum(torch.stack(log_probs) * returns) # torch.stack to concat list of tensors
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        done = False

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # emissions = tracker.stop()
    # writer.add_scalar("emissions", emissions, args.total_timesteps)

    envs.close()
    writer.close()
