import torch
from clip import load, tokenize
from clip.model import VisionTransformer, ModifiedResNet
import copy


class MyCLIP(torch.nn.Module):
    def __init__(
        self,
        name,
        device="cpu",
        use_float=False,
        dataset=None,
        VPT_info=None,
        method=None,
        method_args=None,
    ):
        super().__init__()
        self.clip, preprocess = load(name, device)
        preprocess.transforms = [preprocess.transforms[0], preprocess.transforms[-1]]
        self.preprocess = preprocess

        self.method = method
        self.optim_modules = {}
        self.optim_params = {}

        # optim_modules amd optim_params
        self.image_prompt = None
        self.token_prompt = None
        if VPT_info:
            if VPT_info["image_prompt"]:
                self.image_prompt = torch.nn.Parameter(
                    torch.randn(
                        [1, 3, VPT_info["image_size"], VPT_info["image_size"]], device=device
                    )
                )
                self.optim_params["image_prompt"] = self.image_prompt
            if VPT_info["token_prompt"] > 0:
                self.token_prompt = torch.nn.Parameter(
                    torch.randn(
                        [1, VPT_info["token_prompt"], 768],
                        dtype=self.clip.dtype,
                        device=device,
                    )
                )
                self.optim_params["token_prompt"] = self.token_prompt
        else:
            self.optim_modules["clip.visual"] = self.clip.visual

        self.dtype_dict = {}
        for n, p in self.named_parameters():
            self.dtype_dict[n] = p.dtype

        if use_float:
            self.float()

        # text_features
        if self.method in [
            "TeCoA",
            "PMG_AFT",
            "TGA_ZSR",
            "AGFT",
        ]:
            with torch.no_grad():
                texts = tokenize(dataset.clip_prompts).to(device)
                self.text_features = self.encode_text(texts)
        else:
            print("unknown method!")

        # save
        if self.method in ["TeCoA"]:
            pass
        elif self.method in ["PMG_AFT"]:
            self.visual_orig = copy.deepcopy(self.clip.visual)
        elif self.method in ["AGFT"]:
            self.visual_orig = copy.deepcopy(self.clip.visual)
            self.logit_scale_train = method_args[self.method]["logit_scale"]
        elif self.method in ["TGA_ZSR"]:
            self.visual_orig = copy.deepcopy(self.clip.visual)
        else:
            print("unknown method!")

        self.mode = "val"

        self.requires_grad_(False)
        torch.cuda.empty_cache()

    def restore_dtype(self):
        for n, p in self.named_parameters():
            if self.dtype_dict[n] != torch.float32:
                p.data = p.data.to(self.dtype_dict[n])
                if p.grad:
                    p.grad.data = p.grad.data.to(self.dtype_dict[n])

    def get_optim_parameters(self):
        param_list = []
        for module in self.optim_modules.values():
            param_list.extend(list(module.parameters()))
        param_list.extend(list(self.optim_params.values()))
        return param_list

    def state_dict_train(self):
        dict = {"modules": {}, "params": {}}
        for name, module in self.optim_modules.items():
            dict["modules"][name] = module.state_dict()
        for name, param in self.optim_params.items():
            dict["params"][name] = param.data
        return dict

    def load_state_dict_train(self, dict):
        for name, state in dict["modules"].items():
            self.get_submodule(name).load_state_dict(state)
        for name, data in dict["params"].items():
            param = self.get_parameter(name)
            param.data = data.to(param.data.device)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.optim_modules.values():
            module.requires_grad_(mode)
        for param in self.optim_params.values():
            param.requires_grad_(mode)

    def set_mode(self, mode: str):
        self.mode = mode

    #############################################################
    def clip_visual_foward(self, visual: torch.nn.Module, x: torch.Tensor, orig=False):
        if isinstance(visual, VisionTransformer):
            if self.image_prompt is not None and not orig:
                x = x + self.image_prompt
            x = visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    visual.class_embedding.to(x.dtype)
                    + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + visual.positional_embedding.to(x.dtype)

            if self.token_prompt is not None and not orig:
                x = torch.cat(
                    [
                        x,
                        self.token_prompt
                        + torch.zeros(
                            x.shape[0],
                            self.token_prompt.shape[1],
                            x.shape[-1],
                            dtype=x.dtype,
                            device=x.device,
                        ),
                    ],
                    dim=1,
                )

            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            if self.method in ["TGA_ZSR"]:
                x = visual.ln_post(x)
            else:
                x = visual.ln_post(x[:, 0, :])
            if visual.proj is not None:
                x = x @ visual.proj
            return x
        elif isinstance(visual, ModifiedResNet):
            if self.method in ["TGA_ZSR"]:
                print("Cannot perform TGA_ZSR on ModifiedResNet")
            if self.token_prompt is not None and not orig:
                print("Cannot add token_prompt on ModifiedResNet")
            if self.image_prompt is not None and not orig:
                x = x + self.image_prompt
            x = x.type(visual.conv1.weight.dtype)
            x = visual.relu1(visual.bn1(visual.conv1(x)))
            x = visual.relu2(visual.bn2(visual.conv2(x)))
            x = visual.relu3(visual.bn3(visual.conv3(x)))
            x = visual.avgpool(x)
            x = visual.layer1(x)
            x = visual.layer2(x)
            x = visual.layer3(x)
            x = visual.layer4(x)
            x = visual.attnpool(x)
            return x
        else:
            print("Unknown visual")

    def encode_image(self, image):
        return self.clip_visual_foward(self.clip.visual, image.type(self.clip.dtype))

    def encode_image_orig(self, image):
        return self.clip_visual_foward(self.visual_orig, image.type(self.clip.dtype), orig=True)

    def encode_text(self, text):
        return self.clip.encode_text(text)

    def cos_sim(self, image_features, text_features, logit_scale=None):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        if not logit_scale:
            logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image

    def forward(self, image, text=None):
        if self.method in ["TeCoA", "PMG_AFT"]:
            image = self.preprocess(image)
            image_features = self.encode_image(image)
            return self.cos_sim(image_features, self.text_features)
        elif self.method in ["AGFT"]:
            image = self.preprocess(image)
            image_features = self.encode_image(image)
            if self.mode == "train":
                return self.cos_sim(image_features, self.text_features, self.logit_scale_train)
            elif self.mode == "val":
                return self.cos_sim(image_features, self.text_features)
            else:
                print("unknown mode!")
        elif self.method in ["TGA_ZSR"]:
            image = self.preprocess(image)
            image_features = self.encode_image(image)[:, 0, :]
            return self.cos_sim(image_features, self.text_features)
        else:
            image = self.preprocess(image)
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            return self.cos_sim(image_features, text_features)
