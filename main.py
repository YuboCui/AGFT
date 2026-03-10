import torch
import argparse, time, os, random, numpy

from train import train
from validate import validate


def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--time", type=str, default=time.strftime("%Y%m%d-%H%M%S", time.localtime())
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_or_val", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="TeCoA",
        choices=["TeCoA", "PMG_AFT", "TGA_ZSR", "AGFT"],
    )
    parser.add_argument(
        "--method_args",
        default='{"PMG_AFT": {"alpha": 1, "beta": 1}, "TGA_ZSR": {"alpha": 0.08, "beta": 0.05}, "AGFT": {"logit_scale": 100, "scale_ratio": 1}}',
    )
    parser.add_argument("--VPT", action="store_true")
    parser.add_argument("--image_prompt", action="store_true")
    parser.add_argument("--token_prompt", type=int, default=0)

    # model
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_float", action="store_true")
    parser.add_argument("--enable_amp", action="store_true")

    # data
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--root", type=str, default="../dataset")
    parser.add_argument(
        "--imagenet_root", type=str, default="../dataset/ImageNet"
    )
    parser.add_argument("--train_dataset", type=str, default="ImageNet")
    parser.add_argument(
        "--val_dataset",
        type=str,
        default=[
            "Caltech101",
            "Caltech256",
            "CIFAR10",
            "CIFAR100",
            # "Country211",
            "DTD",
            "EuroSAT",
            "FGVCAircraft",
            "Flowers102",
            "Food101",
            "ImageNet",
            "OxfordIIITPet",
            "PCAM",
            "StanfordCars",
            "STL10",
            "SUN397",
            # "TinyImageNet",
            # "ImageNet_R",
            # "ImageNet_Sketch",
        ],
    )

    # attack
    parser.add_argument("--train_eps", type=str, default="1/255")
    parser.add_argument("--train_alpha", type=str, default="1/255")
    parser.add_argument("--train_steps", type=int, default=2)
    parser.add_argument(
        "--val_attack",
        type=str,
        default="None",
        choices=[
            "None",
            "PGD",
            "CW",
            "AA",
            "PGD_t_rand",
            "PGD_t_least",
            "FGSM",
            "MIFGSM",
            "DIFGSM",
            "TIFGSM",
            "NIFGSM",
            "PIFGSM",
            "PIFGSMPP",
            "MIFGSM_t",
            "DIFGSM_t",
            "TIFGSM_t",
            "NIFGSM_t",
        ],
    )
    parser.add_argument("--val_eps", type=str, default="1/255")
    parser.add_argument("--val_alpha", type=str, default="1/255")
    parser.add_argument("--val_steps", type=int, default=20)

    # train
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--weight_path", type=str, default=None)
    parser.add_argument("--clip_grad", action="store_true")

    # optimization
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-5)  # 1e-5 for batch 256, 1 GPU
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.95))
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_t", type=float, default=1000)  # (1400 for batch 128, 1 GPU)

    args = parser.parse_args()
    args.train_eps = eval(args.train_eps)
    args.train_alpha = eval(args.train_alpha)
    args.val_eps = eval(args.val_eps)
    args.val_alpha = eval(args.val_alpha)
    args.method_args = eval(args.method_args)
    if args.VPT:
        args.VPT_info = {
            "image_prompt": args.image_prompt,
            "image_size": args.image_size,
            "token_prompt": args.token_prompt,
        }
        args.filename = "{}_{}_{}_VPT_{}_{}_{}".format(
            args.time,
            args.model,
            args.method,
            args.image_prompt,
            args.token_prompt,
            args.train_dataset,
        ).replace("/", "_")
    else:
        args.VPT_info = None
        args.filename = "{}_{}_{}_{}".format(
            args.time, args.model, args.method, args.train_dataset
        ).replace("/", "_")

    return args


def initialize_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    args = parse_option()
    args.local_rank, args.world_size, args.device = local_rank, world_size, device
    initialize_seed(args.seed + local_rank)

    if args.weight_path:
        args.filename = args.weight_path[:-4]

    if args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        with open(os.path.join(args.save_dir, args.filename + ".txt"), "a") as f:
            f.write("CLIP " + ("train" if args.train_or_val else "validate") + "\n")
            f.write(str(args) + "\n")
        f.close()

    if args.train_or_val:
        train(args)
    else:
        validate(args)

    torch.distributed.destroy_process_group()
