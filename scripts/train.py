import argparse
import time
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from visual_causal_transformer import VisualCausalTransformer
from working_memory_env.envs.grid_world import DMTSGridEnv


# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
# parser.add_argument("--algo", required=True,
#                     help="algorithm to use: a2c | ppo (REQUIRED)")
# parser.add_argument("--env", required=True,
#                     help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--tile-size", type=int, default=4,
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
# parser.add_argument("--frames-per-proc", type=int, default=None,
#                     help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--device", default="cpu")
parser.add_argument("--suffix", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    device = args.device

    # args.mem = args.recurrence > 1

    # Set run dir

    # date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    # default_model_name = f"{args.grid_size}x{args.grid_size}_{args.tile_size}_seed{args.seed}_{date}"
    default_model_name = f"{args.grid_size}x{args.grid_size}_tile{args.tile_size}_delay{args.max_delay_frames}_frames{args.frames}_dmodel{args.d_model}_nlayers{args.nlayers}_nhead{args.nhead}_seed{args.seed}"
    if args.suffix:
        default_model_name += f'_{args.suffix}'
    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        # envs.append(utils.make_env(args.env, args.seed + 10000 * i))
        envs.append(DMTSGridEnv(
            grid_size=args.grid_size,
            max_delay=args.max_delay_frames,
            tile_size=args.tile_size
        ))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)

    # Load model

    model = VisualCausalTransformer(
        state_dim=2,
        act_dim=envs[0].action_space.n,
        n_blocks=args.nlayers,
        d_model=args.d_model,
        n_heads=args.nhead,
        context_len=args.max_delay_frames+3,
        drop_p=0.5,
        obs_space=obs_space,
    )
    
    model.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(model))

    # Load algo
    algo = torch_ac.DMTSAlgo(
        envs, 
        model, 
        device,
        args.max_delay_frames + 3, 
        args.lr, 
        preprocess_obss=preprocess_obss
    )
    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        train_log = algo.update()
        update_end_time = time.time()

        update += 1
        # tb_writer.add_scalar('Loss', utils.synthesize(train_log['loss']).values(), update)
        tb_writer.add_scalar('Loss', train_log['loss'], update)
        # tb_writer.add_scalar('Accuracy', utils.synthesize(train_log('acc')).values(), update)
        tb_writer.add_scalar('Accuracy', train_log['acc'], update)
        # tb_writer.add_graph(model, train_log["final_input"])
        # tb_writer.add_embedding(train_log["embed"], metadata=train_log["labels"], label_img=train_log["img"])
        num_frames += train_log["num_frames"]
        txt_logger.info(f'Loss: {train_log["loss"]} Acc: {train_log["acc"]}')

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
