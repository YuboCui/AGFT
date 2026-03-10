from torchvision import transforms
from attacks import *

from clip_dataset import *


def get_train_dataset(dataset, root, imagenet_root, image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    train_transform_none = transforms.Compose([transforms.ToTensor()])
    if dataset == "ImageNet":
        return ImageNet(imagenet_root, "train", transform=train_transform)
    elif dataset == "TinyImageNet":
        return TinyImageNet(root, "train", transform=train_transform)
    elif dataset == "CIFAR10":
        return CIFAR10(root, True, transform=train_transform_none, download=True)
    elif dataset == "CIFAR100":
        return CIFAR100(root, True, transform=train_transform_none, download=True)


def get_val_dataset(dataset_list, root, imagenet_root, image_size):
    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
        ]
    )
    val_transform_none = transforms.Compose([transforms.ToTensor()])

    res = []
    for d in dataset_list:
        if d == "Caltech101":
            res.append(Caltech101(root, transform=val_transform, download=True))
        if d == "Caltech256":
            res.append(Caltech256(root, transform=val_transform, download=True))
        if d == "CIFAR10":
            res.append(CIFAR10(root, False, transform=val_transform_none, download=True))
        if d == "CIFAR100":
            res.append(CIFAR100(root, False, transform=val_transform_none, download=True))
        if d == "Country211":
            res.append(Country211(root, "test", transform=val_transform, download=True))
        if d == "DTD":
            res.append(DTD(root, "test", transform=val_transform, download=True))
        if d == "EuroSAT":
            res.append(EuroSAT(root, transform=val_transform, download=True))
        if d == "FGVCAircraft":
            res.append(FGVCAircraft(root, "test", transform=val_transform, download=True))
        if d == "Flowers102":
            res.append(Flowers102(root, "test", transform=val_transform, download=True))
        if d == "Food101":
            res.append(Food101(root, "test", transform=val_transform, download=True))
        if d == "ImageNet":
            res.append(ImageNet(imagenet_root, "val", transform=val_transform))
        if d == "OxfordIIITPet":
            res.append(OxfordIIITPet(root, "test", transform=val_transform, download=True))
        if d == "PCAM":
            res.append(PCAM(root, "test", transform=val_transform, download=True))
        if d == "StanfordCars":
            res.append(StanfordCars(root, "test", transform=val_transform, download=True))
        if d == "STL10":
            res.append(STL10(root, "test", transform=val_transform_none, download=True))
        if d == "SUN397":
            res.append(SUN397(root, transform=val_transform, download=True))
        if d == "TinyImageNet":
            res.append(TinyImageNet(root, "val", transform=val_transform))
        if d == "ImageNet_R":
            res.append(ImageNet_R(root, transform=val_transform))
        if d == "ImageNet_Sketch":
            res.append(ImageNet_Sketch(root, transform=val_transform))
    return res


def get_val_attack(model, attack, eps, alpha, steps):
    if attack == "PGD":
        return PGD(model, eps, alpha, steps)
    elif attack == "PGD_t_rand":
        att = PGD(model, eps, alpha, steps)
        att.set_mode_targeted_random()

    elif attack == "PGD_t_least":
        att = PGD(model, eps, alpha, steps)
        att.set_mode_targeted_least_likely()
        return att
    elif attack == "CW":
        return CW(model, eps, alpha, steps)
    elif attack == "AA":
        return AutoAttack(model, "Linf", eps, "rand")
    elif attack == "FGSM":
        return FGSM(model, eps)
    elif attack == "MIFGSM":
        return MIFGSM(model, eps, alpha, steps)
    elif attack == "MIFGSM_t":
        att = MIFGSM(model, eps, alpha, steps)
        att.set_mode_targeted_random()
        return att
    elif attack == "DIFGSM":
        return DIFGSM(model, eps, alpha, steps)
    elif attack == "DIFGSM_t":
        att = DIFGSM(model, eps, alpha, steps)
        att.set_mode_targeted_random()
        return att
    elif attack == "TIFGSM":
        return TIFGSM(model, eps, alpha, steps)
    elif attack == "TIFGSM_t":
        att = TIFGSM(model, eps, alpha, steps)
        att.set_mode_targeted_random()
        return att
    elif attack == "NIFGSM":
        return NIFGSM(model, eps, alpha, steps)
    elif attack == "NIFGSM_t":
        att = NIFGSM(model, eps, alpha, steps)
        att.set_mode_targeted_random()
        return att
    elif attack == "PIFGSM":
        return PIFGSM(model, eps, steps)
    elif attack == "PIFGSMPP":
        return PIFGSMPP(model, eps, steps)
    elif attack == "PIFGSMPP_t":
        att = PIFGSMPP(model, eps, steps)
        att.set_mode_targeted_random()
        return att
    else:
        return None
