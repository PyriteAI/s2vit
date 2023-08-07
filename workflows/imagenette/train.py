import math
import random
from collections.abc import Callable, Sequence
from typing import Any, cast

import flash_tinyvit as ftv
import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from ema_pytorch import EMA
from lion_pytorch import Lion
from PIL import ImageFilter
from timm.data import Mixup, RandomResizedCropAndInterpolation
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import RepeatAugSampler
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms


class CosineAnnealingLR(optim.lr_scheduler.LambdaLR):
    """Cosine learning rate with warmup.

    Either the number of warmup steps can be specified, or a fraction of the total steps can be specified for the
    warmup. If neither are specified, no warmup is performed. If both are specified, an error is raised.

    Args:
        optimizer: Optimizer.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps. Defaults to None.
        warmup_fraction: Fraction of training steps used to warmup. Defaults to None.
        num_cycles: Number of cycles in the cosine annealing. Defaults to 0.5.
        last_epoch: Last epoch. Defaults to -1.

    Raises:
        ValueError: If both warmup_steps and warmup_fraction are specified.
    """

    @staticmethod
    def _create_lr_lambda(warmup_steps: int, total_steps: int, num_cycles: float):
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return lr_lambda

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        warmup_steps: int | None = None,
        warmup_fraction: float | None = None,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        if warmup_steps is not None and warmup_fraction is not None:
            raise ValueError("only one of warmup_steps and warmup_fraction can be set")
        if warmup_steps is None and warmup_fraction is None:
            warmup_steps = 0
        elif warmup_fraction is not None:
            warmup_steps = round(total_steps * warmup_fraction)
        assert warmup_steps is not None

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        super().__init__(
            optimizer, self._create_lr_lambda(self.warmup_steps, self.total_steps, self.num_cycles), last_epoch
        )


class LightningImagenetteClassifier(pl.LightningModule):
    def __init__(
        self,
        depths: Sequence[int] = (2, 2, 6, 2),
        dims: Sequence[int] = (64, 128, 160, 320),
        patch_sizes: Sequence[int | tuple[int, int]] = (4, 2, 2, 2),
        dim_head: int = 32,
        block_size: int = 8,
        dim_ff: int | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        drop_block_size: int = 7,
        bias: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.0, num_classes=10)
        self.model = ftv.FlashTinyBlockViT(
            depths=depths,
            dims=dims,
            patch_sizes=patch_sizes,
            global_pool=True,
            num_classes=10,
            dim_head=dim_head,
            block_size=block_size,
            dim_ff=dim_ff,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_block_rate=drop_block_rate,
            drop_block_size=drop_block_size,
            bias=bias,
        )
        self.ema_model = EMA(
            self.model,
            beta=0.9998,
            update_after_step=100,
            update_every=1,
            inv_gamma=1.0,
            power=0.75,
            include_online_model=False,
        )

    def training_step(self, batch: Any, batch_idx: int):
        images, targets = batch["image"], batch["label"]
        images, targets = self.mixup(images, targets)
        preds = self.model(images)
        loss = F.binary_cross_entropy_with_logits(preds, targets)

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.ema_model.update()

    def validation_step(self, batch: Any, batch_idx: int):
        images, targets = batch["image"], batch["label"]
        preds = self.ema_model(images)
        loss = F.binary_cross_entropy_with_logits(preds, F.one_hot(targets, num_classes=preds.shape[1]).float())

        acc = (preds.argmax(dim=1) == targets).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> Any:
        optimizer = Lion(self.model.parameters(), lr=1.0e-5, weight_decay=1.0e-3)
        lr_scheduler = CosineAnnealingLR(
            optimizer, total_steps=int(self.trainer.estimated_stepping_batches), warmup_steps=2000
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}

    def train_dataloader(self):
        dataset = create_train_dataset("frgfm/imagenette", name="full_size", split_name="train")
        dataset = cast(Dataset, dataset)
        transform = create_train_transform(image_size=256)
        dataset.set_transform(apply_transform(transform))

        sampler = RepeatAugSampler(
            dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, num_repeats=4
        )
        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=64,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = create_val_dataset("frgfm/imagenette", name="full_size", split_name="validation")
        dataset = cast(Dataset, dataset)
        transform = create_val_transform(image_size=256)
        dataset.set_transform(apply_transform(transform))

        sampler = DistributedSampler(
            dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=False  # type: ignore
        )
        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=64,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader


def create_train_transform(
    image_size: int,
    p_random_erasing: float = 0.0,
    random_erasing_scale: tuple[float, float] = (0.02, 0.33),
    random_erasing_ratio: tuple[float, float] = (0.3, 3.3),
):
    t = transforms.Compose(
        [
            RandomResizedCropAndInterpolation(image_size, interpolation="random"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice(
                [
                    transforms.Grayscale(3),
                    transforms.RandomSolarize(128, p=1.0),
                    transforms.Lambda(lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))),
                ],
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            # NOTE: Grayscale seems to ignore `num_output_channels` if the input is a PIL Image, so we need to expand
            # the tensor to 3 channels manually.
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            transforms.RandomErasing(
                p=p_random_erasing, scale=tuple(random_erasing_scale), ratio=tuple(random_erasing_ratio)
            ),
        ]
    )
    return t


def create_val_transform(image_size: int, ratio: float = 1.0):
    image_resize = int(image_size / ratio)
    t = transforms.Compose(
        [
            transforms.Resize(image_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # NOTE: Grayscale seems to ignore `num_output_channels` if the input is a PIL Image, so we need to expand
            # the tensor to 3 channels manually.
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return t


def apply_transform(transform: Callable):
    def apply_transform_fn(data: dict[str, Any]):
        data["image"] = [transform(img) for img in data["image"]]
        return data

    return apply_transform_fn


def create_train_dataset(path: str, split_name: str = "train", **kwargs: Any):
    dataset = load_dataset(path, split=split_name, **kwargs)
    return dataset


def create_val_dataset(path: str, split_name: str = "validation", **kwargs: Any):
    dataset = load_dataset(path, split=split_name, **kwargs)
    return dataset


def main():
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)

    model = LightningImagenetteClassifier()
    trainer = pl.Trainer(
        precision="bf16-mixed",
        max_epochs=200,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(monitor="val_acc", patience=10, mode="max"),
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}", monitor="val_acc", save_last=True
            ),
        ],
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
