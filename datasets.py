import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


mean = {
    "mnist": [0.1307],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4867, 0.4408],
}
std = {
    "mnist": [0.3081],
    "cifar10": [0.2023, 0.1994, 0.2010],
    "cifar100": [0.2675, 0.2565, 0.2761],
}


class data_processor(nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.args = args
        self.mean = torch.tensor(mean[self.args.dataset], device=device)
        self.std = torch.tensor(std[self.args.dataset], device=device)
        self.alpha = 0.95

    def preprocess(self, x: torch.tensor) -> torch.tensor:
        if self.args.uniform_dequantize:
            x = uniform_dequantize(x)
        if self.args.gaussian_dequantize:
            x = x + torch.randn_like(x) * 0.01
        if self.args.data_centered:
            x = 2 * x - 1.0
        elif self.args.logit_transform:
            x = self.alpha + (1 - 2 * self.alpha) * x
            x = (x / (1 - x)).log()

        if self.args.normalize_image:
            x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        return x

    def postprocess(self, x: torch.tensor) -> torch.tensor:
        if self.args.logit_transform:
            x = (x.sigmoid() - self.alpha) / (1 - 2 * self.alpha)
        elif self.args.data_centered:
            x = x * 0.5 + 0.5

        if self.args.normalize_image:
            x = x * self.std[None, :, None, None] + self.mean[None, :, None, None]
        x = x.clamp(min=0.0, max=1.0) if self.args.clamp else x

        return x


def uniform_dequantize(
    x: torch.tensor,
    nvals: int = 256,
) -> torch.tensor:
    """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
    noise = x.new().resize_as_(x).uniform_()
    x = x * (nvals - 1) + noise
    x = x / nvals
    return x


def get_dataset(args):
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.workers
    print("Using batch size {}".format(BATCH_SIZE))
    print("Using {} number of workers".format(NUM_WORKERS))
    train_transforms = [transforms.Resize(args.img_size), transforms.ToTensor()]
    test_transforms = [transforms.Resize(32), transforms.ToTensor()]

    if args.horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    if args.dataset == "mnist":
        dataset = torchvision.datasets.MNIST
    elif args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif args.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100

    trainset = dataset(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(train_transforms),
    )
    testset = dataset(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(test_transforms),
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return trainloader, testloader
