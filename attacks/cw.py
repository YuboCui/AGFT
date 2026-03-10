import torch

from torchattacks.attack import Attack


class CW(Attack):
    # Linf
    def __init__(self, model, eps=8 / 255, alpha=0.8 / 255, steps=30, kappa=0, random_start=True):
        super().__init__("CW", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.kappa = kappa
        self.random_start = random_start

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            cost = -self.f(outputs, labels).mean()

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]), device=outputs.device)[labels]

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit

        return torch.clamp((j - i), min=-self.kappa)
