import argparse
import numpy

import utils
from utils import device
from working_memory_env.envs.grid_world import DMTSGridEnv


# Parse arguments

parser = argparse.ArgumentParser()
# parser.add_argument("--env", required=True,
#                     help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--tile-size", type=int, default=4,
                    help="size of each cell in term of pixels")
parser.add_argument("--grid-size", type=int, default=4,
                    help="square grid size")
parser.add_argument("--d_model", type=int, default=10,
                    help="transformer embedding size")
parser.add_argument("--nhead", type=int, default=1,
                    help="transformer attention heads")
parser.add_argument("--nlayers", type=int, default=2,
                    help="transformer MLP layers")
parser.add_argument("--max-delay-frames", type=int, default=5,
                    help="maximum number of delay frames per episode")

args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
print(f"Device: {device}\n")

# Load environment

# env = utils.make_env(args.env, args.seed, render_mode="human")
env = DMTSGridEnv(
        grid_size=args.grid_size,
        max_delay=args.max_delay_frames,
        tile_size=args.tile_size,
        render_mode="human"
    )
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir, args)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif
    frames = []

# Create a window to view the environment
env.render()

for episode in range(args.episodes):
    obs, _ = env.reset()

    while True:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        action = agent.get_action(obs, env.step_count)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        # agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            agent.reset_()
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
