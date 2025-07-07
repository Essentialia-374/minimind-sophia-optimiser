#!/usr/bin/env python3

import os
import sys
import argparse
import math
import time
import warnings
from contextlib import nullcontext

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings("ignore")

# Sophia‑G
class SophiaG(optim.Optimizer):
    r"""
    这是 Sophia 论文中 Sophia-G 变体的 PyTorch 实现。该变体使用高斯-牛顿-巴特利特（GNB）估算器，不过在此实现中，我使用平均梯度的平方作为近似
    Sophia arxiv: https://arxiv.org/abs/2305.14342
    对于一个参数张量 θ，在第 t 步的更新公式（逐元素计算）如下：
        
        mₜ   = β₁ · mₜ₋₁ + (1−β₁)·g
        hₜ   = β₂ · hₜ₋ₖ + (1−β₂)·ĝ       (其中 k = update_freq)
        u    = clip( mₜ / max(γ·hₜ, ε), ρ )
        θ   ← θ − η · u  −  η·λ·θ          (解耦权重衰减)
        
    其中 clip 是指将每个坐标（元素）的值裁剪到 `±ρ` 的范围内。
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.96, 0.99),
        eps: float = 1e-12,
        rho: float = 0.05,
        weight_decay: float = 0.0,
        gamma: float = 0.01,
        update_freq: int = 10,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < betas[0] < 1.0:
            raise ValueError("Invalid beta1 value")
        if not 0.0 < betas[1] < 1.0:
            raise ValueError("Invalid beta2 value")
        if eps <= 0.0:
            raise ValueError("Invalid eps value")
        if rho <= 0.0:
            raise ValueError("Invalid rho value")
        if gamma <= 0.0:
            raise ValueError("Invalid gamma value")
        if update_freq < 1:
            raise ValueError("update_freq must be ≥ 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            rho=rho,
            weight_decay=weight_decay,
            gamma=gamma,
            update_freq=update_freq,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            rho = group["rho"]
            wd = group["weight_decay"]
            gamma = group["gamma"]
            k = group["update_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"]      = 0
                    state["exp_avg"]   = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg     = state["exp_avg"]
                exp_hessian = state["exp_hessian"]
                state["step"] += 1
                step = state["step"]

                # 动量项 mₜ
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 每 k 步进行一次海森/高斯-牛顿对角线更新。
                if step % k == 0:
                    hess_diag = grad.pow(2) # GNB 近似
                    exp_hessian.mul_(beta2).add_(hess_diag, alpha=1.0 - beta2)

                # 分母项：γ·h 或 ε
                denom = exp_hessian.mul(gamma).clamp_min(eps)

                # 预条件更新 + 裁剪
                update = exp_avg / denom
                update.clamp_(min=-rho, max=rho)

                if wd != 0.0:
                    p.data.add_(p.data, alpha=-lr * wd)  # 解耦权重衰减
                p.data.add_(update, alpha=-lr)

        return loss

# helper
def Logger(msg: str):
    if (not ddp) or dist.get_rank() == 0:
        print(msg)


def get_lr(current_step: int, total_steps: int, base_lr: float) -> float:
    return base_lr / 10 + 0.5 * base_lr * (1 + math.cos(math.pi * current_step / total_steps))


# training loop
def train_epoch(epoch: int, wandb_run):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask  = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step,
                    args.epochs * iter_per_epoch,
                    args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with ctx:
            res  = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = (loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        # gradient accumulation
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # logging
        if step % args.log_interval == 0:
            spent = time.time() - start_time
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}] "
                f"({step}/{iter_per_epoch}) "
                f"loss:{loss.item() * args.accumulation_steps:.3f} "
                f"lr:{optimizer.param_groups[-1]['lr']:.12f} "
                f"epoch_Time:{spent / (step + 1) * iter_per_epoch / 60 - spent / 60:.1f} min"
            )
            if (wandb_run is not None) and ((not ddp) or dist.get_rank() == 0):
                wandb_run.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spent / (step + 1) * iter_per_epoch / 60 - spent / 60,
                    }
                )

        # checkpoint
        if ((step + 1) % args.save_interval == 0) and ((not ddp) or dist.get_rank() == 0):
            model.eval()
            moe_tag = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_tag}.pth"

            state_dict = (
                model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            )
            state_dict = {k: v.half() for k, v in state_dict.items()}  # half precision
            torch.save(state_dict, ckp)
            model.train()


# set‑up helpers
def init_model(cfg: MiniMindConfig):
    tokenizer = AutoTokenizer.from_pretrained("../model/")
    mdl = MiniMindForCausalLM(cfg).to(args.device)
    Logger(f"模型参数大小: {sum(p.numel() for p in mdl.parameters() if p.requires_grad) / 1e6:.2f} M")
    return mdl, tokenizer


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind – Pre‑training with Sophia‑G")
    # generic training args
    parser.add_argument("--out_dir",        type=str,   default="../out")
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--learning_rate",  type=float, default=5e-4)
    parser.add_argument("--device",         type=str,   default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",          type=str,   default="bfloat16")
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--wandb_project",  type=str,   default="MiniMind-Pretrain")
    parser.add_argument("--num_workers",    type=int,   default=1)
    parser.add_argument("--ddp",            action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--log_interval",   type=int,   default=100)
    parser.add_argument("--save_interval",  type=int,   default=100)
    # model‑specific
    parser.add_argument("--hidden_size",        type=int, default=512)
    parser.add_argument("--num_hidden_layers",  type=int, default=8)
    parser.add_argument("--max_seq_len",        type=int, default=512)
    parser.add_argument("--use_moe",            type=bool, default=False)
    parser.add_argument("--data_path",          type=str, default="../dataset/pretrain_hq.jsonl")
    # Sophia‑G hyper‑parameters
    parser.add_argument("--optimizer_beta1",      type=float, default=0.96)
    parser.add_argument("--optimizer_beta2",      type=float, default=0.99)
    parser.add_argument("--optimizer_rho",        type=float, default=0.05)
    parser.add_argument("--optimizer_gamma",      type=float, default=0.01)
    parser.add_argument("--optimizer_eps",        type=float, default=1e-12)
    parser.add_argument("--optimizer_update_freq",type=int,   default=10)
    parser.add_argument("--weight_decay",         type=float, default=0.1)
    args = parser.parse_args()

    # derived paths
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # random seed for reproducibility
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # ddp initialisation
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank = 0
    DEVICE = "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        torch.manual_seed(base_seed + dist.get_rank())
        torch.cuda.manual_seed(base_seed + dist.get_rank())

    # wandb
    if args.use_wandb and ((not ddp) or ddp_local_rank == 0):
        import wandb

        run_name = f"MiniMind-Pretrain-SophiaG-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=run_name)
    else:
        wandb = None

    # build model / data
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    model, tokenizer = init_model(lm_config)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))

    # Sophia‑G 优化器
    optimizer = SophiaG(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.optimizer_beta1, args.optimizer_beta2),
        rho=args.optimizer_rho,
        gamma=args.optimizer_gamma,
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
        update_freq=args.optimizer_update_freq,
    )

    # ddp
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # training 
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    iter_per_epoch = len(train_loader)

    for ep in range(args.epochs):
        train_epoch(ep, wandb)
