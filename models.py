import timm
import torch
from torchvision.models import (
    vit_b_16, resnet152, swin_s,
    ViT_B_16_Weights, ResNet152_Weights, Swin_S_Weights,
)
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model


def get_classifier_head(model):
    """Get the classifier head module for different model architectures."""
    model_name = model.__class__.__name__

    if model_name == "VisionTransformer":
        return model.heads
    elif model_name == "SwinTransformer":
        return model.head
    elif model_name == "ResNet":
        return model.fc
    elif hasattr(model, 'head'):
        return model.head
    elif model_name == "CLIPClassifier":
        return model.head
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def apply_training_strategy_1_head_only(model):
    """Strategy 1: Train only the classifier head (last layer)."""
    for param in model.parameters():
        param.requires_grad = False

    head = get_classifier_head(model)
    for param in head.parameters():
        param.requires_grad = True

    print(f"[Strategy 1] Training only classifier head: {get_trainable_params_info(model)}")
    return model


def apply_training_strategy_2_partial(model, percentage=0.3):
    """Strategy 2: Train last percentage% of layers."""
    for param in model.parameters():
        param.requires_grad = False

    named_params = list(model.named_parameters())
    total_layers = len(named_params)

    num_to_unfreeze = max(1, int(total_layers * percentage))

    for name, param in named_params[-num_to_unfreeze:]:
        param.requires_grad = True

    print(f"[Strategy 2] Training last {percentage*100}% of layers ({num_to_unfreeze}/{total_layers}): {get_trainable_params_info(model)}")
    return model


def apply_training_strategy_3_full(model):
    """Strategy 3: Train all weights."""
    for param in model.parameters():
        param.requires_grad = True

    print(f"[Strategy 3] Training all weights: {get_trainable_params_info(model)}")
    return model


def apply_training_strategy_4_lora(model):
    """Strategy 4: Train with LoRA (Low-Rank Adaptation)."""
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    print(f"[Strategy 4] Training with LoRA: {get_trainable_params_info(model)}")
    return model


def get_trainable_params_info(model):
    """Get information about trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0
    return f"{trainable:,} / {total:,} ({percentage:.2f}%)"


def apply_training_strategy(model, strategy: str):
    """
    Apply a training strategy to the model.

    Args:
        model: PyTorch model
        strategy: One of 'head_only', 'partial', 'full', 'lora'

    Returns:
        Modified model with the selected training strategy
    """
    if strategy == "head_only":
        return apply_training_strategy_1_head_only(model)
    elif strategy == "partial":
        return apply_training_strategy_2_partial(model, percentage=0.3)
    elif strategy == "full":
        return apply_training_strategy_3_full(model)
    elif strategy == "lora":
        return apply_training_strategy_4_lora(model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: head_only, partial, full, lora")


class CLIPClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.vision_model = self.clip.vision_model

        embed_dim = self.clip.config.vision_config.hidden_size

        self.head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        vision_outputs = self.vision_model(pixel_values=x)
        image_embeds = vision_outputs.pooler_output
        logits = self.head(image_embeds)
        return logits


VIT_MODEL = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
SWIN_MODEL = swin_s(weights=Swin_S_Weights.DEFAULT)
RESNET_MODEL = resnet152(weights=ResNet152_Weights.DEFAULT)
DINO_MODEL = timm.create_model("vit_base_patch16_dinov3", pretrained=True)

in_features = DINO_MODEL.num_features

DINO_MODEL.head = torch.nn.Sequential(
    torch.nn.Linear(in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 2),
)


VIT_MODEL.heads = torch.nn.Sequential(
    torch.nn.Linear(in_features=VIT_MODEL.heads.head.in_features, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(in_features=256, out_features=2),
)
SWIN_MODEL.head = torch.nn.Sequential(
    torch.nn.Linear(in_features=SWIN_MODEL.head.in_features, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(in_features=256, out_features=2),
)
RESNET_MODEL.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=RESNET_MODEL.fc.in_features, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(in_features=256, out_features=2),
)

CLIP_MODEL = CLIPClassifier(num_classes=2)

models = [
    VIT_MODEL,
    SWIN_MODEL,
    RESNET_MODEL,
    DINO_MODEL,
    CLIP_MODEL
]


def get_model_parameters(_model):
    return sum(p.numel() for p in _model.parameters() if p.requires_grad)


def get_model_size(_model):
    param_size = 0
    for param in _model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in _model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 3
    return 'model size: {:.3f}GB'.format(size_all_mb)


for model in models:
    model_name = model.__class__.__name__
    print(model_name, get_model_parameters(model))
    print(model_name, get_model_size(model))
