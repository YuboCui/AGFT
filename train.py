import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from timm.scheduler.cosine_lr import CosineLRScheduler
import os, logging
from tqdm import tqdm

from clip_model import MyCLIP
from attacks import PGD
from utils import get_train_dataset, get_val_dataset
from validate import validate_one


def train_one_epoch(epoch, model, train_loader, optim, loss_fn, scheduler, attack, scaler, args):
    train_loader.sampler.set_epoch(epoch)
    if args.local_rank == 0:
        train_loader_tqdm = tqdm(train_loader)
    else:
        train_loader_tqdm = train_loader

    num_batches_per_epoch = len(train_loader)

    for idx, (x, y) in enumerate(train_loader_tqdm):
        scheduler.step(num_batches_per_epoch * epoch + idx)
        optim.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=args.enable_amp):
            x, y = x.to(args.device), y.to(args.device)

            if args.method in ["TeCoA"]:
                model.eval()
                x_adv = attack(x, y)
                model.train()
                y_hat = model(x_adv)
                loss = loss_fn(y_hat, y)
            elif args.method in ["PMG_AFT"]:
                # Compared to https://github.com/serendipity1122/Pre-trained-Model-Guided-Fine-Tuning-for-Zero-Shot-Adversarial-Robustness
                # we followed the original paper
                loss_CE, loss_KL = loss_fn[0], loss_fn[1]
                model.eval()
                x_adv = attack(x, y)
                model.train()
                with torch.no_grad():
                    image_features_orig_adv = model.module.encode_image_orig(
                        model.module.preprocess(x_adv)
                    )
                    y_hat_orig_adv = model.module.cos_sim(
                        image_features_orig_adv,
                        model.module.text_features,
                    )
                y_hat_adv = model(x_adv)
                y_hat_clean = model(x)
                loss = (
                    loss_CE(y_hat_adv, y)
                    + args.method_args[args.method]["alpha"]
                    * loss_KL(y_hat_adv.log_softmax(dim=1), y_hat_orig_adv.softmax(dim=1))
                    + args.method_args[args.method]["beta"]
                    * loss_KL(y_hat_adv.log_softmax(dim=1), y_hat_clean.softmax(dim=1))
                )
            elif args.method in ["TGA_ZSR"]:

                def attention_map(image_features, text_features_truth):
                    # https://github.com/zhyblue424/TGA-ZSR
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    img_spatial_feat = image_features[:, 1:, :]
                    am = img_spatial_feat @ text_features_truth.unsqueeze(-1)
                    am = (am - am.min(1, keepdim=True)[0]) / (
                        am.max(1, keepdim=True)[0] - am.min(1, keepdim=True)[0]
                    )
                    side = int(am.shape[1] ** 0.5)
                    am = am.reshape(am.shape[0], side, side, -1).permute(0, 3, 1, 2)
                    am = torch.nn.functional.interpolate(am, args.image_size, mode="bilinear")
                    return am

                with torch.no_grad():
                    text_features_truth = model.module.text_features[y]
                    image_features_orig = model.module.encode_image_orig(model.module.preprocess(x))
                    am_orig = attention_map(image_features_orig, text_features_truth)
                model.eval()
                x_adv = attack(x, y)
                model.train()

                image_features_clean = model.module.encode_image(model.module.preprocess(x))
                if model.module.token_prompt is not None:
                    image_features_clean = image_features_clean[
                        :, : image_features_orig.shape[1], :
                    ]
                am_clean = attention_map(image_features_clean, text_features_truth)
                loss_AR = torch.mean(torch.norm(am_clean - am_orig, dim=1, p=2))

                image_features_adv = model.module.encode_image(model.module.preprocess(x_adv))
                if model.module.token_prompt is not None:
                    image_features_adv = image_features_adv[:, : image_features_orig.shape[1], :]
                am_adv = attention_map(image_features_adv, text_features_truth)
                loss_AMC = torch.mean(torch.norm(am_adv - am_orig, dim=1, p=2))

                y_hat = model.module.cos_sim(
                    image_features_adv[:, 0, :], model.module.text_features
                )
                loss = (
                    loss_fn(y_hat, y)
                    + args.method_args[args.method]["alpha"] * loss_AR
                    + args.method_args[args.method]["beta"] * loss_AMC
                )
            elif args.method in ["AGFT"]:
                with torch.no_grad():
                    image_features_orig = model.module.encode_image_orig(model.module.preprocess(x))
                    y_hat_orig = model.module.cos_sim(
                        image_features_orig,
                        model.module.text_features,
                        args.method_args[args.method]["scale_ratio"]
                        * model.module.logit_scale_train,
                    )
                    y = y_hat_orig.softmax(dim=1)
                model.eval()
                x_adv = attack(x, y)
                model.train()
                y_hat = model(x_adv)
                loss = loss_fn(y_hat, y)
            else:
                print("unknown method!")

            scaler.scale(loss).backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.module.get_optim_parameters(), args.clip_grad)
            scaler.step(optim)
        scaler.update()


def train(args):
    train_data = get_train_dataset(
        args.train_dataset, args.root, args.imagenet_root, args.image_size
    )
    val_data = get_val_dataset(
        [args.train_dataset], args.root, args.imagenet_root, args.image_size
    )[0]

    train_sampler = DistributedSampler(train_data, seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
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
        train_data,
        args.VPT_info,
        args.method,
        args.method_args,
    )
    if "RN" in args.model:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    if args.optimizer == "sgd":
        optim = torch.optim.SGD(
            model.module.get_optim_parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optim = torch.optim.AdamW(
            model.module.get_optim_parameters(),
            lr=args.lr,
            betas=args.betas,
            weight_decay=args.weight_decay,
        )

    scheduler = CosineLRScheduler(
        optim, t_initial=len(train_loader) * args.epoch, warmup_t=args.warmup_t
    )

    loss_train = torch.nn.CrossEntropyLoss()
    loss_val = torch.nn.CrossEntropyLoss(reduction="sum")

    attack_train = PGD(model, args.train_eps, args.train_alpha, args.train_steps)
    attack_val = PGD(model, args.train_eps, args.train_alpha, args.train_steps)

    if args.method in ["PMG_AFT"]:
        loss_train = [loss_train, torch.nn.KLDivLoss(reduction="batchmean")]

    scaler = torch.amp.GradScaler(enabled=args.enable_amp)

    if args.local_rank == 0:
        logging.basicConfig(
            filename=os.path.join(args.save_dir, args.filename + ".txt"),
            filemode="a",
            format="%(asctime)s  %(message)s",
            datefmt="%Y%m%d %H:%M:%S",
            level=logging.INFO,
        )

    for i in range(args.epoch):
        model.train()
        model.module.set_mode("train")
        train_one_epoch(
            i, model, train_loader, optim, loss_train, scheduler, attack_train, scaler, args
        )
        model.module.set_mode("val")
        model.eval()
        val_loss, val_acc = validate_one(model, val_loader, loss_val, attack_val, args)
        if args.local_rank == 0:
            print("loss: {}, acc: {}".format(val_loss, val_acc))
            logging.info("epoch: {}, val_loss: {}, val_acc: {}".format(i, val_loss, val_acc))

        if (i + 1) % args.save_freq == 0 or (i + 1) == args.epoch:
            if args.local_rank == 0:
                save_path = os.path.join(args.save_dir, args.filename + "_" + str(i + 1) + ".pth")
                torch.save(model.module.state_dict_train(), save_path)


if __name__ == "__main__":
    pass
