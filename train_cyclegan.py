from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from imggen import CycleGANModel, UnpairedImageDataset, save_image_tensor, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CycleGAN for photo-to-art style generation.")
    parser.add_argument("--train-a", type=str, required=True, help="Domain A folder, usually real photos.")
    parser.add_argument("--train-b", type=str, required=True, help="Domain B folder, usually paintings.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name, e.g. monet2photo_custom.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-every-epoch", type=int, default=1, help="Checkpoint save interval.")
    parser.add_argument("--sample-every-epoch", type=int, default=1, help="Preview export interval.")
    parser.add_argument("--resume", type=str, default="", help="Optional training checkpoint to resume from.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP on CUDA.")

    # 超参数：输入图片大小。
    parser.add_argument("--image-size", type=int, default=256, help="Training crop size.")
    # 超参数：单次放入显存图像的个数
    parser.add_argument("--batch-size", type=int, default=1, help="Mini-batch size.")
    # 超参数：训练次数
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs.")
    # 超参数：学习率
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Initial Adam learning rate.")
    # 超参数：0.5 是经典 GAN 的 beta1。
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    # 超参数：标准的循环一致性权重，想更强地保留原图内容时可以适当调大。
    parser.add_argument("--lambda-cycle", type=float, default=10.0, help="Cycle consistency weight.")
    # 超参数：0.5 表示 identity loss 使用 0.5 * lambda_cycle，如果颜色漂移过大可以适当调高。
    parser.add_argument("--lambda-identity", type=float, default=0.5, help="Identity mapping weight factor.")
    # 超参数：64 是标准通道数
    parser.add_argument("--generator-channels", type=int, default=64, help="Base generator channels.")
    parser.add_argument("--discriminator-channels", type=int, default=64, help="Base discriminator channels.")
    # 超参数：256x256 训练一般 9 个残差块
    parser.add_argument("--res-blocks", type=int, default=9, help="Generator residual blocks.")
    # 超参数：官方 CycleGAN 默认使用 instance norm，建议先不要改。
    parser.add_argument("--norm-type", type=str, default="instance", choices=["instance", "batch", "none"], help="Normalization used by official-style generators/discriminators.")
    # 超参数：官方 CycleGAN 默认 no_dropout=True。风格变化不明显时先不要盲目打开 dropout。
    parser.add_argument("--use-dropout", action="store_true", help="Enable dropout in the official-style generator.")
    # 超参数：PatchGAN 判别器层数，官方 basic 判别器等价于 3 层。
    parser.add_argument("--discriminator-layers", type=int, default=3, help="Number of PatchGAN layers.")
    # 超参数：官方仓库默认 normal 初始化。
    parser.add_argument("--init-type", type=str, default="normal", choices=["normal", "xavier", "kaiming", "orthogonal"], help="Weight initialization method.")
    parser.add_argument("--init-gain", type=float, default=0.02, help="Initialization scaling gain.")
    # 超参数：伪样本缓存池大小
    parser.add_argument("--pool-size", type=int, default=50, help="Historical fake-image pool size.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_preview_batch(
    model: CycleGANModel,
    sample_batch: dict[str, object],
    device: torch.device,
    preview_dir: Path,
    epoch: int,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        real_a = sample_batch["A"].to(device)  # type: ignore[assignment]
        real_b = sample_batch["B"].to(device)  # type: ignore[assignment]
        fake_b = model.netG_A(real_a)
        fake_a = model.netG_B(real_b)
        save_image_tensor(real_a, preview_dir / f"epoch_{epoch:03d}_real_a.png")
        save_image_tensor(fake_b, preview_dir / f"epoch_{epoch:03d}_fake_b.png")
        save_image_tensor(real_b, preview_dir / f"epoch_{epoch:03d}_real_b.png")
        save_image_tensor(fake_a, preview_dir / f"epoch_{epoch:03d}_fake_a.png")
    model.train()


def make_lr_lambda(total_epochs: int):
    def lr_lambda(epoch_index: int) -> float:
        half = total_epochs // 2
        if epoch_index < half:
            return 1.0
        remaining = max(1, total_epochs - half)
        return max(0.0, 1.0 - (epoch_index - half) / remaining)

    return lr_lambda


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    dataset = UnpairedImageDataset(
        root_a=args.train_a,
        root_b=args.train_b,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CycleGANModel(
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        generator_channels=args.generator_channels,
        discriminator_channels=args.discriminator_channels,
        res_blocks=args.res_blocks,
        pool_size=args.pool_size,
        norm_type=args.norm_type,
        use_dropout=args.use_dropout,
        discriminator_layers=args.discriminator_layers,
        init_type=args.init_type,
        init_gain=args.init_gain,
    ).to(device)
    optimizer_g = torch.optim.Adam(
        list(model.netG_A.parameters()) + list(model.netG_B.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )
    optimizer_d = torch.optim.Adam(
        list(model.netD_A.parameters()) + list(model.netD_B.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=make_lr_lambda(args.epochs))
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=make_lr_lambda(args.epochs))
    scaler_g = GradScaler(enabled=args.mixed_precision and device.type == "cuda")
    scaler_d = GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    start_epoch = 1
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        start_epoch = int(checkpoint["epoch"]) + 1

    metadata = {
        "experiment_name": args.experiment_name,
        "created_at": datetime.now().isoformat(),
        "train_a": args.train_a,
        "train_b": args.train_b,
        "num_domain_a": len(dataset.paths_a),
        "num_domain_b": len(dataset.paths_b),
        "device": str(device),
        "args": vars(args),
    }
    save_json(metadata, output_dir / "config.json")

    preview_sample = next(iter(dataloader))
    for epoch in range(start_epoch, args.epochs + 1):
        progress = tqdm(dataloader, desc=f"epoch {epoch}/{args.epochs}", ncols=120)
        for batch in progress:
            real_a = batch["A"].to(device, non_blocking=True)  # type: ignore[assignment]
            real_b = batch["B"].to(device, non_blocking=True)  # type: ignore[assignment]

            model.set_requires_grad([model.netD_A, model.netD_B], False)
            optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=args.mixed_precision and device.type == "cuda"):
                generated, losses = model.generator_loss(real_a, real_b)
            scaler_g.scale(losses.loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            model.set_requires_grad([model.netD_A, model.netD_B], True)
            optimizer_d.zero_grad(set_to_none=True)
            fake_b_for_d = model.fake_b_pool.query(generated["fake_b"])
            fake_a_for_d = model.fake_a_pool.query(generated["fake_a"])
            with autocast(enabled=args.mixed_precision and device.type == "cuda"):
                loss_d_a = model.discriminator_loss(model.netD_A, real_b, fake_b_for_d)
                loss_d_b = model.discriminator_loss(model.netD_B, real_a, fake_a_for_d)
                loss_d = loss_d_a + loss_d_b
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            progress.set_postfix(
                loss_g=f"{losses.loss_g.item():.3f}",
                loss_gan=f"{losses.loss_gan.item():.3f}",
                loss_cycle=f"{losses.loss_cycle.item():.3f}",
                loss_idt=f"{losses.loss_idt.item():.3f}",
                loss_d_a=f"{loss_d_a.item():.3f}",
                loss_d_b=f"{loss_d_b.item():.3f}",
            )

        scheduler_g.step()
        scheduler_d.step()

        if epoch % args.sample_every_epoch == 0:
            save_preview_batch(model, preview_sample, device, output_dir / "previews", epoch)

        if epoch % args.save_every_epoch == 0:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "config": vars(args),
            }
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.ckpt")
            torch.save(model.make_checkpoint(), output_dir / f"epoch_{epoch:03d}_generators.pth")

    torch.save(model.make_checkpoint(), output_dir / "final_generators.pth")
    print(f"Training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
