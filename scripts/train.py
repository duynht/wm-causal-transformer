import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from model import ACModel
from visual_transformer import CausalVisionTransformer
from working_memory_env.envs.grid_world import DMTSGridEnv


# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
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
parser.add_argument("--grid-size", type=int, default=256,
                    help="square grid size")
parser.add_argument("d_model", type=int, default=10,
                    help="transformer embedding size")
parser.add_argument("nlayers", type=int, default=2,
                    help="transformer MLP layers")
parser.add_argument("max-delay-frames", type=int, default=5,
                    help="maximum number of delay frames per episode")
# parser.add_argument("--frames-per-proc", type=int, default=None,
#                     help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--device", default="cpu")

if __name__ == "__main__":
    args = parser.parse_args()

    device = args.device

    # args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.grid_size}x{args.grid_size}_{args.algo}_seed{args.seed}_{date}"

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

    # acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    # if "model_state" in status:
    #     acmodel.load_state_dict(status["model_state"])
    acmodel = CausalVisionTransformer(
        in_channels=obs_space["image"][2],
        out_channels=32,
        kernel_size=3,
        obs_space=obs_space,
        action_space=envs[0].action_space,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_model,
        nlayers=args.nlayers,
        max_len=args.max_delay_frames+3
    )
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == 'dmts':
        algo = torch_ac.DMTSAlgo(
            envs, 
            acmodel, 
            device,
            args.max_delay_frames + 3, 
            args.lr, 
            preprocess_obss=preprocess_obss
        )

    # elif args.algo =='pure_supervised':


    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

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
        # exps, logs1 = algo.collect_experiences()
        # logs2 = algo.update_parameters(exps)
        # logs = {**logs1, **logs2}
        train_log = algo.update()
        update_end_time = time.time()

        update += 1
        # tb_writer.add_scalar('Loss', utils.synthesize(train_log['loss']).values(), update)
        tb_writer.add_scalar('Loss', train_log['loss'], update)
        # tb_writer.add_scalar('Accuracy', utils.synthesize(train_log('acc')).values(), update)
        tb_writer.add_scalar('Accuracy', train_log['acc'], update)
        num_frames += train_log["num_frames"]

        # Print logs

        # if update % args.log_interval == 0:
        #     fps = logs["num_frames"] / (update_end_time - update_start_time)
        #     duration = int(time.time() - start_time)
        #     return_per_episode = utils.synthesize(logs["return_per_episode"])
        #     rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        #     num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        #     # header = ["update", "frames", "FPS", "duration"]
        #     # data = [update, num_frames, fps, duration]
        #     # header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        #     # data += rreturn_per_episode.values()
        #     # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        #     # data += num_frames_per_episode.values()
        #     # header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        #     # data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        #     txt_logger.info(
        #         "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
        #         .format(*data))

        #     header += ["return_" + key for key in return_per_episode.keys()]
        #     data += return_per_episode.values()

        #     if status["num_frames"] == 0:
        #         csv_logger.writerow(header)
        #     csv_logger.writerow(data)
        #     csv_file.flush()

        #     for field, value in zip(header, data):
        #         tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
