"""Base trainer to train diffusion models 
"""
import time
import torch
import torch.optim as optim
import torch.utils.data as data
from accelerate import Accelerator
from torchvision.utils import save_image

import numpy as np
from ..sde import get_sde
from ..models import get_model

from .registry import register
from ..utils import AverageMeter, get_lr
from datasets import data_processor


class BaseTrainer:
    def __init__(self, args) -> None:

        self.args = args
        self.accelerator = Accelerator(fp16=args.fp16)
        self.device = self.accelerator.device
        self.reduce_op = (
            torch.mean
            if args.reduce_mean
            else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        )
        self.data_processor = data_processor(args, self.device)

        # get score function and SDE
        self.sde = get_sde(args, self.device)
        self.score_function = get_model(args, self.device)

        # pptimizer and schedulers
        self.optimizer = optim.Adam(
            self.score_function.parameters(),
            lr=args.lr,
            eps=args.eps,
        )

        # loss meters
        self.step = 1
        self.eval_loss = AverageMeter()
        self.train_loss = AverageMeter()
        self.train_nll = AverageMeter()
        self.cnf_train_nfe = AverageMeter()

        # time meters
        self.data_time = AverageMeter()
        self.train_time = AverageMeter()

    def lr_schedule(self, step):
        if self.args.warmup > 0:
            for g in self.optimizer.param_groups:
                g["lr"] = self.args.lr * np.minimum(step / self.args.warmup, 1.0)

    def forward_pass(self, data: torch.tensor, is_train: bool = True) -> torch.tensor:
        if self.args.likelihood_weighting:
            pass
        else:
            # get time steps
            time_steps = torch.rand(data.shape[0], device=self.device)
            time_steps = (
                time_steps * (1.0 - self.args.train_time_eps) + self.args.train_time_eps
            )
            # get parameters of marginal distribution
            mean_t, var_t, std_t = self.sde.marginal_distribution(data, time_steps)
            weights = var_t

        # Sample from marginal distribution
        noise = torch.randn_like(data)
        perturbed_data = mean_t + std_t * noise
        # get scores to denoise the perturbed data
        scores = self.score_function(perturbed_data, time_steps)
        # calulate weights for the loss function
        losses = torch.square(scores + noise / std_t) * weights
        losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)

        return loss

    def evaluate(self, testloader: data.DataLoader) -> None:
        self.score_function.eval()
        with torch.no_grad():
            for idx, (data, _) in enumerate(testloader):
                # preprocess
                data = self.data_processor.preprocess(data)
                eval_loss = self.forward_pass(data, is_train=False)
                self.eval_loss.update(eval_loss.item(), data.shape[0])
                if idx % self.args.log_step == 0:
                    print(
                        "Eval Step: {} "
                        "Eval Loss: {:.4f}".format(idx, self.eval_loss.avg)
                    )
        print("Eval Loss: {:.4f}".format(self.eval_loss.avg))
        print("================================")

    def train(self, trainloader: data.DataLoader, testloader: data.DataLoader) -> None:
        # Prepare the model, dataloaders and opt
        (
            self.score_function,
            self.optimizer,
            trainloader,
            testloader,
        ) = self.accelerator.prepare(
            self.score_function,
            self.optimizer,
            trainloader,
            testloader,
        )

        iterator = iter(trainloader)
        start_time = time.time()
        for step in range(self.step, self.args.train_steps + 1):
            try:
                data, _ = next(iterator)
            except:
                iterator = iter(trainloader)
                data, _ = next(iterator)

            # log data loading time
            self.data_time.update(time.time() - start_time)

            # preprocess
            data = self.data_processor.preprocess(data)
            self.score_function.train()
            self.score_function.zero_grad(set_to_none=True)
            # train step
            train_loss = self.forward_pass(data, is_train=True)
            # backward pass
            self.accelerator.backward(train_loss)
            # lr schedule
            # self.lr_schedule(step)
            self.optimizer.step()
            # log train step time and loss
            self.train_time.update(time.time() - start_time)
            self.train_loss.update(train_loss.item(), data.shape[0])

            # save the network and other config
            if step % self.args.save_step == 0:
                pass
            if step % self.args.log_step == 0:
                print(
                    "Train Step: {} Train Loss: {:.4f} Train Loss avg: {:.4f} "
                    "Learning Rate: {:.4f} CNF NFE: {} CNF NFE AVG: {}".format(
                        step,
                        self.train_loss.val,
                        self.train_loss.avg,
                        get_lr(self.optimizer),
                        self.cnf_train_nfe.val,
                        self.cnf_train_nfe.avg,
                    )
                )
            if step % self.args.sample_step == 0:
                self.sample(step)

    def sample(self, step) -> None:

        print("Sampling from Reverse SDE")
        self.score_function.eval()
        with torch.no_grad():
            sample_shape = (
                self.args.num_samples,
                self.args.num_channels,
                self.args.img_size,
                self.args.img_size,
            )
            sample, nfe, ode_solve_time = self.sde.sample(
                self.score_function, sample_shape
            )

        print("NFE: {} ODE solve time: {:.4f}".format(nfe, ode_solve_time))
        save_image(
            self.data_processor.postprocess(sample),
            "./samples/sample_{}.png".format(step),
        )


@register
def base_trainer(args):
    return BaseTrainer(args)
