import os
import shutil
import kagglehub
import torchvision
from kagglehub.config import set_kaggle_credentials

from dataset import Dataset

# Label map
ID_TO_LABEL = {
    0: "Baked Potato",
    1: "Taco"
}

ROOT = os.path.join(os.getcwd(), "data", "1", "Food Classification dataset")

def download_remote_and_move_into_project():
    set_kaggle_credentials(
        username=os.getenv("KAGGLE_USERNAME", "your_username"),
        api_key=os.getenv("KAGGLE_KEY", "your_api_key")
    )

    if not os.path.exists("./data"):
        path = kagglehub.dataset_download("harishkumardatalab/food-image-classification-dataset")
        os.makedirs("./data", exist_ok=True)
        shutil.move(path, "./data")


    for _dir in os.scandir(ROOT):
        if _dir.name not in ["Taco", "Baked Potato"]:
            shutil.rmtree(_dir.path)

def get_dataset(image_size: int):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.6),
        torchvision.transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ) # ImageNet stats
    ])

    dataset = Dataset(ROOT, transform=transforms)
    return dataset
