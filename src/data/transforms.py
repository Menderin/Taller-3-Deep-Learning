"""Image transformations shared by training and inference."""

from __future__ import annotations

from torchvision import transforms


class TransformFactory:
    """Create deterministic and augmented torchvision pipelines."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    @classmethod
    def training(cls, image_size: int, use_augmentation: bool = True):
        operations = [transforms.Resize((image_size, image_size))]
        if use_augmentation:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                    ),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
            ]
        )
        return transforms.Compose(operations)

    @classmethod
    def evaluation(cls, image_size: int):
        """Validation, test and inference must not include randomness."""

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
            ]
        )

    @classmethod
    def inference(cls, image_size: int):
        return cls.evaluation(image_size)
