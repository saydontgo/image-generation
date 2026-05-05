from __future__ import annotations

from dataclasses import dataclass
import functools
import re
from typing import Callable

import torch
from torch import nn

from .image_pool import ImagePool


class ResnetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, n_blocks: int = 9) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        current_dim = ngf
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(current_dim, current_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(current_dim * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            current_dim *= 2

        for _ in range(n_blocks):
            layers.append(ResnetBlock(current_dim))

        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_dim,
                        current_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(current_dim // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            current_dim //= 2

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(current_dim, output_nc, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_official_norm_layer(norm_type: str = "instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda _channels: nn.Identity()
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


class OfficialResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer: Callable[[int], nn.Module],
        use_dropout: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        conv_block: list[nn.Module] = []
        padding = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            padding = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True),
            ]
        )
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        padding = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            padding = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                norm_layer(dim),
            ]
        )
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class OfficialResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_layer: Callable[[int], nn.Module] | None = None,
        use_dropout: bool = False,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = get_official_norm_layer("instance")
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            layers.extend(
                [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            )

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            layers.append(
                OfficialResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            )

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        (ngf * mult) // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer((ngf * mult) // 2),
                    nn.ReLU(True),
                ]
            )

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def patch_official_instance_norm_state_dict(state_dict: dict[str, torch.Tensor], module: nn.Module, keys: list[str], i: int = 0) -> None:
    key = keys[i]
    if i + 1 == len(keys):
        if module.__class__.__name__.startswith("InstanceNorm") and (key == "running_mean" or key == "running_var"):
            if getattr(module, key) is None:
                state_dict.pop(".".join(keys), None)
        if module.__class__.__name__.startswith("InstanceNorm") and key == "num_batches_tracked":
            state_dict.pop(".".join(keys), None)
    else:
        patch_official_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def build_official_generator_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    norm_type: str = "instance",
    use_dropout: bool = False,
) -> nn.Module:
    if "model.1.weight" not in state_dict:
        raise KeyError("Unsupported official CycleGAN checkpoint: missing model.1.weight")

    ngf = int(state_dict["model.1.weight"].shape[0])
    block_indices = {
        int(match.group(1))
        for key in state_dict
        for match in [re.match(r"model\.(\d+)\.conv_block\.1\.weight", key)]
        if match is not None
    }
    if not block_indices:
        raise KeyError("Unsupported official CycleGAN checkpoint: no conv_block weights found")

    norm_layer = get_official_norm_layer(norm_type)
    generator = OfficialResnetGenerator(
        ngf=ngf,
        n_blocks=len(block_indices),
        norm_layer=norm_layer,
        use_dropout=use_dropout,
    )
    patched_state_dict = dict(state_dict)
    if hasattr(patched_state_dict, "_metadata"):
        del patched_state_dict._metadata
    for key in list(patched_state_dict.keys()):
        patch_official_instance_norm_state_dict(patched_state_dict, generator, key.split("."))
    generator.load_state_dict(patched_state_dict)
    return generator


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        kw = 4
        padw = 1
        layers: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        current_dim = ndf
        for layer_index in range(1, n_layers):
            next_dim = min(current_dim * 2, 512)
            stride = 1 if layer_index == n_layers - 1 else 2
            layers.extend(
                [
                    nn.Conv2d(current_dim, next_dim, kernel_size=kw, stride=stride, padding=padw, bias=False),
                    nn.InstanceNorm2d(next_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_dim = next_dim

        layers.append(nn.Conv2d(current_dim, 1, kernel_size=kw, stride=1, padding=padw))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


@dataclass
class CycleGANLosses:
    loss_g: torch.Tensor
    loss_d_a: torch.Tensor
    loss_d_b: torch.Tensor
    loss_idt: torch.Tensor
    loss_cycle: torch.Tensor
    loss_gan: torch.Tensor


class CycleGANModel(nn.Module):
    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        res_blocks: int = 9,
        pool_size: int = 50,
    ) -> None:
        super().__init__()
        self.netG_A = ResnetGenerator(ngf=generator_channels, n_blocks=res_blocks)
        self.netG_B = ResnetGenerator(ngf=generator_channels, n_blocks=res_blocks)
        self.netD_A = NLayerDiscriminator(ndf=discriminator_channels)
        self.netD_B = NLayerDiscriminator(ndf=discriminator_channels)

        self.netG_A.apply(init_weights)
        self.netG_B.apply(init_weights)
        self.netD_A.apply(init_weights)
        self.netD_B.apply(init_weights)

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.fake_a_pool = ImagePool(pool_size)
        self.fake_b_pool = ImagePool(pool_size)
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def set_requires_grad(self, networks: list[nn.Module], requires_grad: bool) -> None:
        for network in networks:
            for parameter in network.parameters():
                parameter.requires_grad = requires_grad

    def forward_generators(self, real_a: torch.Tensor, real_b: torch.Tensor) -> dict[str, torch.Tensor]:
        fake_b = self.netG_A(real_a)
        rec_a = self.netG_B(fake_b)
        fake_a = self.netG_B(real_b)
        rec_b = self.netG_A(fake_a)
        idt_a = self.netG_B(real_a)
        idt_b = self.netG_A(real_b)
        return {
            "fake_b": fake_b,
            "rec_a": rec_a,
            "fake_a": fake_a,
            "rec_b": rec_b,
            "idt_a": idt_a,
            "idt_b": idt_b,
        }

    def generator_loss(self, real_a: torch.Tensor, real_b: torch.Tensor) -> tuple[dict[str, torch.Tensor], CycleGANLosses]:
        generated = self.forward_generators(real_a, real_b)
        pred_fake_b = self.netD_A(generated["fake_b"])
        pred_fake_a = self.netD_B(generated["fake_a"])
        target_real_b = torch.ones_like(pred_fake_b)
        target_real_a = torch.ones_like(pred_fake_a)
        loss_gan_a = self.criterion_gan(pred_fake_b, target_real_b)
        loss_gan_b = self.criterion_gan(pred_fake_a, target_real_a)
        loss_gan = loss_gan_a + loss_gan_b

        loss_cycle_a = self.criterion_cycle(generated["rec_a"], real_a)
        loss_cycle_b = self.criterion_cycle(generated["rec_b"], real_b)
        loss_cycle = (loss_cycle_a + loss_cycle_b) * self.lambda_cycle

        loss_idt_a = self.criterion_identity(generated["idt_a"], real_a)
        loss_idt_b = self.criterion_identity(generated["idt_b"], real_b)
        loss_idt = (loss_idt_a + loss_idt_b) * self.lambda_cycle * self.lambda_identity

        loss_g = loss_gan + loss_cycle + loss_idt
        losses = CycleGANLosses(
            loss_g=loss_g,
            loss_d_a=torch.zeros((), device=real_a.device),
            loss_d_b=torch.zeros((), device=real_a.device),
            loss_idt=loss_idt,
            loss_cycle=loss_cycle,
            loss_gan=loss_gan,
        )
        return generated, losses

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> torch.Tensor:
        pred_real = discriminator(real_images)
        target_real = torch.ones_like(pred_real)
        loss_real = self.criterion_gan(pred_real, target_real)

        pred_fake = discriminator(fake_images.detach())
        target_fake = torch.zeros_like(pred_fake)
        loss_fake = self.criterion_gan(pred_fake, target_fake)
        return 0.5 * (loss_real + loss_fake)

    def make_checkpoint(self) -> dict[str, object]:
        return {
            "netG_A": self.netG_A.state_dict(),
            "netG_B": self.netG_B.state_dict(),
            "netD_A": self.netD_A.state_dict(),
            "netD_B": self.netD_B.state_dict(),
            "lambda_cycle": self.lambda_cycle,
            "lambda_identity": self.lambda_identity,
        }

    def load_generators_only(self, checkpoint: dict[str, object]) -> None:
        if "netG_A" in checkpoint and "netG_B" in checkpoint:
            self.netG_A.load_state_dict(checkpoint["netG_A"])  # type: ignore[arg-type]
            self.netG_B.load_state_dict(checkpoint["netG_B"])  # type: ignore[arg-type]
            return
        if "G_A" in checkpoint and "G_B" in checkpoint:
            self.netG_A.load_state_dict(checkpoint["G_A"])  # type: ignore[arg-type]
            self.netG_B.load_state_dict(checkpoint["G_B"])  # type: ignore[arg-type]
            return
        raise KeyError("Checkpoint must contain generator weights under netG_A/netG_B or G_A/G_B.")
