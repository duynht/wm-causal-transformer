import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device
from working_memory_env.envs.grid_world import DMTSGridEnv
from icecream import ic

# Parse arguments

parser = argparse.ArgumentParser()
# parser.add_argument("--env", required=True,
#                     help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--tile-size", type=int, default=16,
                    help="size of each cell in term of pixels")
parser.add_argument("--grid-size", type=int, default=4,
                    help="square grid size")
parser.add_argument("--d_model", type=int, default=10,
                    help="transformer embedding size")
parser.add_argument("--nlayers", type=int, default=2,
                    help="transformer MLP layers")
parser.add_argument("--nhead", type=int, default=1,
                    help="transformer attention heads")
parser.add_argument("--max-delay-frames", type=int, default=5,
                    help="maximum number of delay frames per episode")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    print(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        # env = utils.make_env(args.env, args.seed + 10000 * i)
        # envs.append(env)
        envs.append(DMTSGridEnv(
            grid_size=args.grid_size,
            max_delay=args.max_delay_frames,
            tile_size=args.tile_size,
        ))
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir, args, num_envs=args.procs)
    print("Agent loaded\n")

    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()
    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    step = 0
    acc = 0
    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss, step)
        step += 1
        obss, rewards, terminateds, truncateds, _ = env.step(actions, )
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask
        acc += torch.tensor(rewards).sum().item() 
    # acc = agent.analyze_feedbacks(rewards, dones)
    acc /= args.procs
    # breakpoint()
    print(acc)
    end_time = time.time()