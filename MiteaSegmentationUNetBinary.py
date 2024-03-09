"""
Ref: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
"""
import os
import random
from glob import glob

import nrrd
import pytorch_lightning as pl
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses.dice import *
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureType, AsDiscrete
from monai.utils import set_determinism

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(4)   # to reproduce same results


def check_path_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class MiteaSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)

        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(1, 1, 1),
            num_res_units=3,
            norm=Norm.BATCH,
        ).to(device)

        self.loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)

        self.train_metric_function = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.val_metric_function = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.test_metric_function = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        self.train_hausdorff_distance = HausdorffDistanceMetric()
        self.test_hausdorff_distance = HausdorffDistanceMetric()

        self.test_mean_iou = MeanIoU()

        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4, amsgrad=True)

        self.post_pred = Compose(
            [EnsureType("tensor", device=torch.device("cuda") if torch.cuda.is_available() else "cpu"),
             AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose(
            [EnsureType("tensor", device=torch.device("cuda") if torch.cuda.is_available() else "cpu"),
             AsDiscrete(to_onehot=2)])

        self.val_losses_on_step = []
        self.test_losses_on_step = []

        self.best_val_metric = 0
        self.best_val_epoch = 0
        self.best_test_metric = 0

    check_path_exists(os.getcwd() + "/unet_predictions_test")

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        # Create dictionaries for train, val and test files
        train_files_images = sorted(glob(os.path.join("groomed_data/train/", "images", "*.nrrd")))
        train_files_labels = sorted(glob(os.path.join("groomed_data/train/", "labels", "*.nrrd")))
        self.train_files = [{"image": image, "label": label} for image, label in
                            zip(train_files_images, train_files_labels)]

        val_files_images = sorted(glob(os.path.join("groomed_data/val/", "images", "*.nrrd")))
        val_files_labels = sorted(glob(os.path.join("groomed_data/val/", "labels", "*.nrrd")))
        self.val_files = [{"image": image, "label": label} for image, label in
                          zip(val_files_images, val_files_labels)]

        test_files_images = sorted(glob(os.path.join("groomed_data/test/", "images", "*.nrrd")))
        test_files_labels = sorted(glob(os.path.join("groomed_data/test/", "labels", "*.nrrd")))
        self.test_files = [{"image": image, "label": label} for image, label in
                           zip(test_files_images, test_files_labels)]

        # Set deterministic training for reproducibility
        set_determinism(seed=0)

        # Define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
            ]
        )

        # Define datasets
        self.train_dataset = Dataset(data=self.train_files, transform=train_transforms)
        self.val_dataset = Dataset(data=self.val_files, transform=val_transforms)
        self.test_dataset = Dataset(data=self.test_files, transform=test_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=1, num_workers=2)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=2)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=2)
        return test_loader

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"].to(device), batch["label"].to(device)

        # Only epicardium
        labels[labels > 1] = 1

        # Get the dice loss
        output = self.forward(images)
        dice_loss = self.loss_function(output, labels)
        self.log("train_dice_loss", dice_loss.item(), on_step=False, on_epoch=True)

        loss = dice_loss
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)

        # Train dice metric
        post_processed_output = [self.post_pred(pred) for pred in decollate_batch(output)]
        post_processed_labels = [self.post_label(label) for label in decollate_batch(labels)]

        self.train_metric_function(y_pred=post_processed_output, y=post_processed_labels)
        self.train_hausdorff_distance(y_pred=post_processed_output, y=post_processed_labels)

        return {"loss": loss}

    def on_train_epoch_end(self):
        train_metric_epoch = self.train_metric_function.aggregate().item()
        self.train_metric_function.reset()

        self.log("train_dice_metric_epoch", train_metric_epoch, on_step=False, on_epoch=True)

        train_hausdorff_distance = self.train_hausdorff_distance.aggregate().item()
        self.train_hausdorff_distance.reset()

        self.log("train_hausdorff_distance", train_hausdorff_distance, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        images_val, labels_val = batch["image"].to(device), batch["label"].to(device)

        # Only epicardium
        labels_val[labels_val > 1] = 1

        # Get the dice loss
        outputs_val = self.forward(images_val)
        dice_loss = self.loss_function(outputs_val, labels_val)
        self.log("val_dice_loss", dice_loss.item(), on_step=True, on_epoch=True)

        loss = dice_loss

        # Val dice metric
        outputs_val = [self.post_pred(i) for i in decollate_batch(outputs_val)]
        labels_val = [self.post_label(i) for i in decollate_batch(labels_val)]
        self.val_metric_function(y_pred=outputs_val, y=labels_val)

        self.log("val_loss_step", loss.item(), on_step=True, on_epoch=False)
        self.val_losses_on_step.append({"val_loss": loss})

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        val_loss_epoch = torch.stack([x["val_loss"] for x in self.val_losses_on_step]).mean()

        val_metric_epoch = self.val_metric_function.aggregate().item()
        self.val_metric_function.reset()

        self.log("val_loss_epoch", val_loss_epoch, on_step=False, on_epoch=True)
        self.log("val_metric_epoch", val_metric_epoch, on_step=False, on_epoch=True)

        if val_metric_epoch > self.best_val_metric:
            self.best_val_metric = val_metric_epoch
            self.best_val_epoch = self.current_epoch

        print(f"Current epoch: {self.current_epoch} Current metric: {val_metric_epoch:.4f}")
        print(f"Best metric: {self.best_val_metric} at epoch: {self.best_val_epoch:.4f}")

        self.val_losses_on_step.clear()

    def test_step(self, batch, batch_idx):
        images_test, labels_test = batch["image"].to(device), batch["label"].to(device)

        # Only epicardium
        labels_test[labels_test > 1] = 1

        outputs_test = self.forward(images_test)

        # Save prediction as image
        input_image_path = self.test_files[batch_idx]['label']
        image_name = input_image_path.split("/")[3].split(".")[0]
        image_number = image_name.split("_")[0][1:]
        image_data, image_header = nrrd.read(input_image_path)

        predictions_numpy = torch.argmax(outputs_test, dim=1).cpu().numpy().astype(np.uint8)[0]
        output_path = os.getcwd() + "/unet_predictions_test/" + f"test_img_{image_number}.nrrd"
        nrrd.write(output_path, predictions_numpy, header=image_header)

        loss = self.loss_function(outputs_test, labels_test)

        outputs_test = [self.post_pred(i) for i in decollate_batch(outputs_test)]
        labels_test = [self.post_label(i) for i in decollate_batch(labels_test)]

        self.test_metric_function(y_pred=outputs_test, y=labels_test)
        self.test_hausdorff_distance(y_pred=outputs_test, y=labels_test)
        self.test_mean_iou(y_pred=outputs_test, y=labels_test)

        self.log("test_loss_step", loss.item(), on_step=True, on_epoch=False)

        self.test_losses_on_step.append({"test_loss": loss})

        return {"test_loss": loss}

    def on_test_epoch_end(self):
        test_loss_epoch = torch.stack([x["test_loss"] for x in self.test_losses_on_step]).mean()

        test_metric_epoch = self.test_metric_function.aggregate().item()
        test_hausdorff_distance = self.test_hausdorff_distance.aggregate().item()
        test_mean_iou = self.test_mean_iou.aggregate().item()

        self.log("test_loss_epoch", test_loss_epoch, on_step=False, on_epoch=True)
        self.log("test_metric_epoch", test_metric_epoch, on_step=False, on_epoch=True)
        self.log("test_hausdorff_epoch", test_hausdorff_distance, on_step=False, on_epoch=True)
        self.log("test_mean_iou", test_mean_iou, on_step=False, on_epoch=True)

        self.test_metric_function.reset()
        self.test_hausdorff_distance.reset()
        self.test_mean_iou.reset()

        if test_metric_epoch > self.best_test_metric:
            self.best_test_metric = test_metric_epoch

        print(f"Best test metric: {self.best_test_metric}")

        self.test_losses_on_step.clear()
