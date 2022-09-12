import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = ""
TEST_DIR = ""
LR = 3e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMG_SIZE = 256
IMG_CHANNELS = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = False
DISC_CHK = "disc.pth.tar"
GEN_CHK = "gen.pth.tar"

both_transforms = A.Compose(
    [
        A.Resize(width=256,height=256)
    ],
    additional_targets={"image0":"image"}
)

transform_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=255.0),
        ToTensorV2()
    ]
)

transform_mask = A.Compose(
    [
        A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=255.0),
        ToTensorV2()
    ]
)
