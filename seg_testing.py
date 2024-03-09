import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from MiteaSegmentationFusionModelEpi import MiteaSegmentation


if __name__ == '__main__':

    # Load the best model
    best_model_checkpoint_path = os.getcwd() + "/checkpoints_results/best_model-v8.ckpt"

    best_model = MiteaSegmentation.load_from_checkpoint(best_model_checkpoint_path)

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    best_model.to(device)
    best_model.eval()

    test_logger = TensorBoardLogger("tensorboard-logs-results", name="unet-segmentation-test")
    test_trainer = pl.Trainer(accelerator="gpu", devices=1, logger=test_logger, log_every_n_steps=1)

    test_trainer.test(best_model)
