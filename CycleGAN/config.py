import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = "./"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.02
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 50
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_M = "./Checkpoints/GEN/genm.pth.tar" 
CHECKPOINT_GEN_P = "./Checkpoints/GEN/genp.pth.tar" 
CHECKPOINT_DISC_M = "./Checkpoints/DISC/discm.pth.tar" 
CHECKPOINT_DISC_P = "./Checkpoints/DISC/discp.pth.tar" 

transforms = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={"image0":"image"},
)

