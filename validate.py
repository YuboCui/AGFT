import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os

from clip_model import MyCLIP
from utils import get_val_dataset, get_val_attack


def validate_one(model, val_loader, loss_fn, attack, args):
    loss = torch.tensor(0.0, device=args.device)
    acc = torch.tensor(0.0, device=args.device)
    num = torch.tensor(0, device=args.device)

    if args.local_rank == 0:
        val_loader_tqdm = tqdm(val_loader)
    else:
        val_loader_tqdm = val_loader

    for x, y in val_loader_tqdm:
        num += len(y)
        with torch.amp.autocast(device_type="cuda", enabled=args.enable_amp):
            x, y = x.to(args.device), y.to(args.device)
            if attack:
                x_adv = attack(x, y)
                y_hat = model(x_adv)
            else:
                y_hat = model(x)
            loss += loss_fn(y_hat, y)
            acc += (y_hat.argmax(dim=1) == y).sum()
            torch.distributed.barrier()

    num_list = [torch.tensor(0, device=args.device) for _ in range(args.world_size)]
    torch.distributed.all_gather(num_list, num)
    num = sum(num_list)

    loss_list = [torch.tensor(0.0, device=args.device) for _ in range(args.world_size)]
    torch.distributed.all_gather(loss_list, loss)
    loss = sum(loss_list) / num

    acc_list = [torch.tensor(0.0, device=args.device) for _ in range(args.world_size)]
    torch.distributed.all_gather(acc_list, acc)
    acc = sum(acc_list) / num

    return loss, acc


def validate(args):
    if args.local_rank == 0:
        with open(os.path.join(args.save_dir, args.filename + ".txt"), "a") as f:
            f.write(str((args.val_attack, args.val_eps, args.val_alpha, args.val_steps)) + "\n")
        f.close()

    val_data_list = get_val_dataset(
        args.val_dataset, args.root, args.imagenet_root, args.image_size
    )
    val_acc_list = []

    for i in range(len(val_data_list)):
        val_data = val_data_list[i]
        val_sampler = DistributedSampler(val_data, seed=args.seed)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = MyCLIP(
            args.model,
            args.device,
            args.use_float,
            val_data,
            args.VPT_info,
            args.method,
            args.method_args,
        )
        if args.weight_path:
            model.load_state_dict_train(
                torch.load(os.path.join(args.save_dir, args.weight_path), weights_only=True)
            )
        model.train()
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

        loss_val = torch.nn.CrossEntropyLoss(reduction="sum")

        attack_val = get_val_attack(
            model, args.val_attack, args.val_eps, args.val_alpha, args.val_steps
        )

        model.eval()
        loss, acc = validate_one(model, val_loader, loss_val, attack_val, args)
        val_acc_list.append(acc)

        if args.local_rank == 0:
            with open(os.path.join(args.save_dir, args.filename + ".txt"), "a") as f:
                f.write("dataset: {}, loss: {}, acc: {}\n".format(args.val_dataset[i], loss, acc))
            f.close()

        if args.local_rank == 0:
            print("dataset: {}, loss: {}, acc: {}".format(args.val_dataset[i], loss, acc))

    if args.local_rank == 0:
        with open(os.path.join(args.save_dir, args.filename + ".txt"), "a") as f:
            f.write("mean: {}\n".format(sum(val_acc_list) / len(val_acc_list)))
        f.close()
