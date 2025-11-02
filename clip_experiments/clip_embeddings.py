import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model


class CLIPEmbeddingClassifier(nn.Module):
    """
    CLIP-based classifier using cosine similarity between image and text embeddings.
    This follows the zero-shot CLIP approach.
    """

    def __init__(self, class_names, templates=None):
        super().__init__()

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        self.class_names = class_names
        self.templates = templates
        self.register_buffer("text_embeddings", self._get_text_embeddings())

    def _get_text_embeddings(self):
        """Pre-compute text embeddings for all class labels using templates."""
        all_embeddings = []

        with torch.no_grad():
            for class_name in self.class_names:
                texts = [template.format(class_name) for template in self.templates]
                inputs = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                inputs = {k: v.to(self.clip.device) for k, v in inputs.items()}

                text_features = self.clip.get_text_features(**inputs)

                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                text_features = text_features.mean(dim=0)

                all_embeddings.append(text_features)

        text_embeddings = torch.stack(all_embeddings)

        return text_embeddings

    def forward(self, pixel_values):
        """
        Forward pass using image-text similarity.

        Args:
            pixel_values: Preprocessed image tensors (B, C, H, W)

        Returns:
            logits: Similarity scores between images and text classes (B, num_classes)
        """
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ self.text_embeddings.T

        return logits

    def update_text_embeddings(self):
        """Update text embeddings after fine-tuning the text encoder."""
        self.text_embeddings = self._get_text_embeddings()



def apply_clip_strategy_zero_shot(model):
    """Strategy 0: Zero-shot (freeze everything)."""
    for param in model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Zero-Shot] No training - all frozen: {trainable:,} / {total:,} (0.00%)")
    return model


def apply_clip_strategy_partial_30(model):
    """Strategy 1: Train last 30% of layers."""
    for param in model.parameters():
        param.requires_grad = False

    named_params = list(model.named_parameters())
    total_layers = len(named_params)

    num_to_unfreeze = max(1, int(total_layers * 0.3))
    for name, param in named_params[-num_to_unfreeze:]:
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    print(f"[Partial 30%] Training last 30% of layers: {trainable:,} / {total:,} ({percentage:.2f}%)")
    return model


def apply_clip_strategy_both_encoders(model):
    """Strategy 2: Train both ImageEncoder + TextEncoder."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Both Encoders] Training ImageEncoder + TextEncoder: {trainable:,} / {total:,} (100.00%)")
    return model


def apply_clip_strategy_image_only(model):
    """Strategy 3: Train ImageEncoder only."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.clip.vision_model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    print(f"[Image Only] Training ImageEncoder only: {trainable:,} / {total:,} ({percentage:.2f}%)")
    return model


def apply_clip_strategy_text_and_image(model):
    """Strategy 4: Train TextEncoder + ImageEncoder (same as both_encoders)."""
    return apply_clip_strategy_both_encoders(model)


def apply_clip_strategy_lora(model):
    """Strategy 5: Train with LoRA."""
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
    )

    model.clip = get_peft_model(model.clip, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    print(f"[LoRA] Training with LoRA (r=64, alpha=64): {trainable:,} / {total:,} ({percentage:.2f}%)")
    return model


def apply_clip_embedding_strategy(model, strategy: str):
    """
    Apply training strategy to CLIP embedding model.

    Args:
        model: CLIPEmbeddingClassifier instance
        strategy: One of 'zero_shot', 'partial_30', 'both_encoders',
                  'image_only', 'text_and_image', 'lora'

    Returns:
        Modified model
    """
    if strategy == "zero_shot":
        return apply_clip_strategy_zero_shot(model)
    elif strategy == "partial_30":
        return apply_clip_strategy_partial_30(model)
    elif strategy == "both_encoders":
        return apply_clip_strategy_both_encoders(model)
    elif strategy == "image_only":
        return apply_clip_strategy_image_only(model)
    elif strategy == "text_and_image":
        return apply_clip_strategy_text_and_image(model)
    elif strategy == "lora":
        return apply_clip_strategy_lora(model)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose from: "
            "zero_shot, partial_30, both_encoders, image_only, text_and_image, lora"
        )


class CLIPDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to preprocess images for CLIP."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if torch.is_tensor(image):
            import torchvision.transforms.functional as F
            image = F.to_pil_image(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        return pixel_values, label