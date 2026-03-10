from torchvision import datasets as td
from torchvision.datasets.utils import download_and_extract_archive, download_url
import os, shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, Sequence
import pycountry


def refine_classname(name: str):
    return name.lower().replace("_", " ").replace("-", " ").replace("/", " ").strip()


def load_folder2name(path="./imagenet_classes_names.txt"):
    dict = {}
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            name = split_name[2]
            id = split_name[0]
            dict[id] = name
            line = f.readline()
    return dict


class Caltech101(td.Caltech101):
    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo of a {}.",
    ) -> None:
        super().__init__(root, target_type, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [refine_classname(refine_classname(label)) for label in self.categories]
        self.clip_classes[1] = "person"
        self.clip_classes = self.clip_classes[1:]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if target > 0:
            target -= 1
        return img, target


class Caltech256(td.Caltech256):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo of a {}.",
    ) -> None:
        super().__init__(root, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [
            refine_classname(refine_classname(label[4:].replace("-101", "")))
            for label in self.categories
        ]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class CIFAR10(td.CIFAR10):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="This is a photo of a {}.",
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [refine_classname(label) for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class CIFAR100(td.CIFAR100):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="This is a photo of a {}.",
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [refine_classname(label) for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class Country211(td.Country211):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo I took in {}.",
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)

        self.prompt_template = prompt_template
        countries = {}
        for country in pycountry.countries:
            countries[country.alpha_2] = country.name
        countries["XK"] = "Kosovo"
        self.clip_classes = [
            countries[label].split(",")[0].replace("-", " ") for label in self.classes
        ]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class DTD(td.DTD):
    def __init__(
        self,
        root: str,
        split: str = "train",
        partition: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A surface with a {} texture.",
    ) -> None:
        super().__init__(root, split, partition, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [refine_classname(label) for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class EuroSAT(td.EuroSAT):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A centered satellite photo of {}.",
    ) -> None:
        super().__init__(root, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [
            "annual crop land",
            "a forest",
            "brushland or shrubland",
            "a highway or a road",
            "industrial buildings",
            "pasture land",
            "permanent crop land",
            "residential buildings",
            "a river",
            "a sea or a lake",
        ]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class FGVCAircraft(td.FGVCAircraft):
    def __init__(
        self,
        root: str,
        split: str = "trainval",
        annotation_level: str = "variant",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo of a {}, a type of aircraft.",
    ) -> None:
        super().__init__(root, split, annotation_level, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = []
        for label in self.classes:
            if label[0] == "7":
                self.clip_classes.append(f"Boeing {label}")
            else:
                self.clip_classes.append(label)
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class Flowers102(td.Flowers102):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo of a {}, a type of flower.",
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [
            "pink primrose",
            "hard leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colts foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink yellow dahlia",
            "cautleya spicata",
            "japanese anemone",
            "black eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert rose",
            "tree mallow",
            "magnolia",
            "cyclamen ",
            "watercress",
            "canna lily",
            "hippeastrum ",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class Food101(td.Food101):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo of {}, a type of food.",
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [refine_classname(label) for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class ImageNet(td.ImageNet):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        prompt_template="This is a photo of a {}.",
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, **kwargs)

        self.prompt_template = prompt_template
        dict = load_folder2name()
        self.clip_classes = [dict[id].replace("_", " ").replace("-", " ") for id in self.wnids]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class OxfordIIITPet(td.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo of a {}, a type of pet.",
    ):
        super().__init__(
            root, split, target_types, transforms, transform, target_transform, download
        )

        self.prompt_template = prompt_template
        self.clip_classes = [refine_classname(label) for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class PCAM(td.PCAM):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="This is a photo of {}.",
    ):
        super().__init__(root, split, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = ["healthy lymph node tissue", "lymph node tumor tissue"]
        self.clip_prompts = [
            "This is a photo of healthy lymph node tissue.",
            "This is a photo of lymph node tumor tissue.",
        ]


class StanfordCars(td.StanfordCars):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="A photo of a {}.",
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [label[:-5].replace("-", " ") for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )


class STL10(td.STL10):
    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="This is a photo of a {}.",
    ) -> None:
        super().__init__(root, split, folds, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [label for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class SUN397(td.SUN397):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template="This is a photo of a {}.",
    ) -> None:
        super().__init__(root, transform, target_transform, download)

        self.prompt_template = prompt_template
        self.clip_classes = [refine_classname(label) for label in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class TinyImageNet(td.ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        prompt_template="This is a photo of a {}.",
        **kwargs: Any,
    ) -> None:
        root_train = os.path.join(root, "tiny-imagenet-200", "train")
        root = os.path.join(root, "tiny-imagenet-200", split)
        if split == "val" and os.listdir(root) == ["val_annotations.txt", "images"]:
            class_ids = os.listdir(root_train)
            for id in class_ids:
                os.makedirs(os.path.join(root, id))
            dict = {}
            with open(os.path.join(root, "val_annotations.txt")) as f:
                line = f.readline()
                while line:
                    split_name = line.strip().split()
                    file_name = split_name[0]
                    class_id = split_name[1]
                    dict[file_name] = class_id
                    line = f.readline()
            for file_name in os.listdir(os.path.join(root, "images")):
                shutil.move(
                    os.path.join(root, "images", file_name),
                    os.path.join(root, dict[file_name]),
                )
            shutil.rmtree(os.path.join(root, "images"))
        super().__init__(root, **kwargs)
        self.prompt_template = prompt_template
        dict = load_folder2name()
        self.clip_classes = [dict[id].replace("_", " ").replace("-", " ") for id in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class ImageNet_R(td.ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        prompt_template="This is a photo of a {}.",
        **kwargs: Any,
    ) -> None:
        super().__init__(os.path.join(root, "imagenet-r"), **kwargs)
        self.prompt_template = prompt_template
        dict = load_folder2name()
        self.clip_classes = [dict[id].replace("_", " ").replace("-", " ") for id in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]


class ImageNet_Sketch(td.ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        prompt_template="This is a photo of a {}.",
        **kwargs: Any,
    ) -> None:
        super().__init__(os.path.join(root, "sketch"), **kwargs)
        self.prompt_template = prompt_template
        dict = load_folder2name()
        self.clip_classes = [dict[id].replace("_", " ").replace("-", " ") for id in self.classes]
        self.clip_prompts = [self.prompt_template.format(label) for label in self.clip_classes]
