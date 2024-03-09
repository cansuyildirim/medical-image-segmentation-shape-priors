import shutil
import shapeworks as sw
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


from MiteaSegmentationFusionModelEpi import MiteaSegmentation


if __name__ == '__main__':

    # Make sure shapeworks library is being used
    print(f"Using shapeworks module from {sw.__file__}")
    swpath = shutil.which("shapeworks")
    print(f"Using shapeworks from {swpath}")

    # Initialize the model
    model = MiteaSegmentation()
    logger = TensorBoardLogger("tensorboard-logs-results", name="unet-segmentation")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric_epoch',  # Monitor validation metric
        mode='max',
        dirpath='checkpoints_results',
        filename='best_model',
        save_top_k=1,
        verbose=True
    )

    if torch.cuda.is_available():
        print("cuda is available")
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=60,
            logger=logger,
            enable_checkpointing=True,
            num_sanity_val_steps=1,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback]
        )
    else:
        print("running on cpu")
        trainer = pl.Trainer(
            max_epochs=25,
            logger=logger,
            enable_checkpointing=True,
            num_sanity_val_steps=1,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback]
        )

    trainer.fit(model)

    print(f"Training is completed, Best_metric: {model.best_val_metric:.4f} " f"at epoch {model.best_val_epoch}")