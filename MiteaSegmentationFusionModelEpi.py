"""
Ref: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
"""
import os
import random
from glob import glob

import nrrd
import pytorch_lightning as pl
import shapeworks as sw
from DeepSSMUtils import model
from DeepSSMUtils.losses import MSE
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

        self.predictions_test_path = os.getcwd() + "/predictions_test/"
        check_path_exists(self.predictions_test_path)

        path = os.getcwd() + "/Output/deep_ssm_mitea"
        # Initialize TL-DeepSSM trained with images
        self.img_network_path = path + "/tl_epi"
        img_config_path = self.img_network_path + "/mitea_deepssm.json"
        self.img_net = model.DeepSSMNet_TLNet(img_config_path)
        # Load the best trained model
        img_model_dir = self.img_network_path + "/mitea_deepssm/"
        best_img_model_path = os.path.join(img_model_dir, 'best_model.torch')
        self.img_net.load_state_dict(torch.load(best_img_model_path))
        # Set the model to evaluation mode
        self.img_net = self.img_net.to(device)
        for param in self.img_net.ImageEncoder.parameters():
            param.requires_grad = False
        self.img_net.eval()

        # Initialize Base-DeepSSM trained with labels
        self.lab_network_path = path + "/base_epi_labels"
        lab_config_path = self.lab_network_path + "/mitea_deepssm.json"
        self.lab_net = model.DeepSSMNet(lab_config_path)
        # Load the best trained model
        lab_model_dir = self.lab_network_path + "/mitea_deepssm/"
        best_lab_model_path = os.path.join(lab_model_dir, 'best_model.torch')
        self.lab_net.load_state_dict(torch.load(best_lab_model_path))
        # Set the model to evaluation mode
        self.lab_net = self.lab_net.to(device)
        for param in self.lab_net.encoder.parameters():
            param.requires_grad = False
        self.lab_net.eval()

    def forward(self, x):
        return self.model(x)

    def get_image_latent_from_deepssm(self, image_path):
        """
        This function finds the latent code of given image using frozen image encoder of TL-DeepSSM
        :param image_path: path to groomed input image
        :return: latent code of groomed input image
        """
        image = sw.Image(image_path).toArray(copy=True, for_viewing=True)

        img = torch.FloatTensor(np.array(image))
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)

        zt, _ = self.img_net.ImageEncoder(img)
        zt.requires_grad = True

        return zt

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

        # Prepare latent codes of groomed ground truth images {image_number: latent code}
        self.ground_truth_latents_epi = {}
        groomed_images_folder = self.img_network_path + "/groomed_images/"

        groomed_images = [i for i in os.listdir(groomed_images_folder) if i.endswith(".nrrd")]
        for groomed_image in groomed_images:
            groomed_image_path = os.path.join(groomed_images_folder, groomed_image)
            groomed_image_number = groomed_image.split("_")[0][1:]  # e.g. 001

            gt_latent_tl = self.get_image_latent_from_deepssm(groomed_image_path)
            self.ground_truth_latents_epi[groomed_image_number] = gt_latent_tl

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

    def get_prediction_latent_from_deepssm(self, predictions):
        """
        This function finds the latent code of given prediction tensor using frozen image encoder of Base-DeepSSM
        :param predictions: prediction mask
        :return: latent code of prediction
        """
        img = predictions.unsqueeze(0).unsqueeze(0)
        zt, _ = self.lab_net.encoder(img)

        return zt

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"].to(device), batch["label"].to(device)

        # Only epicardium prediction
        labels[labels > 1] = 1

        # Get the dice loss
        output = self.forward(images)
        dice_loss = self.loss_function(output, labels)
        self.log("train_dice_loss", dice_loss.item(), on_step=False, on_epoch=True)

        latent_loss = 0
        # Get latent loss after epoch 20
        if self.current_epoch >= 20:
            predictions = output[0][0]

            input_image_path = self.train_files[batch_idx]['image']
            image_name = input_image_path.split("/")[3].split(".")[0]
            image_number = image_name.split("_")[0][1:]

            z_mesh_tl_epi = self.ground_truth_latents_epi[image_number]
            z_prediction_epi = self.get_prediction_latent_from_deepssm(predictions)

            # Loss functions
            mse_epi_tl = MSE(z_prediction_epi, z_mesh_tl_epi)
            mse_epi_rel_tl_loss = MSE(z_prediction_epi, z_mesh_tl_epi) / MSE(z_prediction_epi * 0, z_mesh_tl_epi)

            l1_epi_tl = torch.mean(torch.abs(z_prediction_epi - z_mesh_tl_epi))
            l1_epi_rel_tl_loss = torch.mean(torch.abs(z_prediction_epi - z_mesh_tl_epi)) / torch.mean(
                torch.abs(z_prediction_epi * 0 - z_mesh_tl_epi))

            z_prediction_epi_normalized = torch.nn.functional.log_softmax(z_prediction_epi, dim=1)
            z_mesh_tl_epi_normalized = torch.nn.functional.softmax(z_mesh_tl_epi, dim=1)
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            kldiv_epi_tl = kl_loss(z_prediction_epi_normalized, z_mesh_tl_epi_normalized)

            # Log loss values
            self.log("train_mse_epi_tl", mse_epi_tl.item(), on_step=False, on_epoch=True)
            self.log("train_mse_epi_rel_tl_loss", mse_epi_rel_tl_loss.item(), on_step=False, on_epoch=True)

            self.log("train_l1_epi_tl", l1_epi_tl.item(), on_step=False, on_epoch=True)
            self.log("train_l1_epi_rel_tl_loss", l1_epi_rel_tl_loss.item(), on_step=False, on_epoch=True)

            self.log("train_kldiv_epi_tl", kldiv_epi_tl.item(), on_step=False, on_epoch=True)

            # Set latent loss
            latent_loss = l1_epi_rel_tl_loss
            self.log("train_latent_loss", latent_loss.item(), on_step=False, on_epoch=True)

        # Add the dice loss and latent loss with a weight
        loss = dice_loss + latent_loss
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

        # Only epicardium prediction
        labels_val[labels_val > 1] = 1

        # Get the dice loss
        outputs_val = self.forward(images_val)
        dice_loss = self.loss_function(outputs_val, labels_val)
        self.log("val_dice_loss", dice_loss.item(), on_step=False, on_epoch=True)

        latent_loss = 0
        # Get latent loss after epoch 20
        if self.current_epoch >= 20:
            predictions_val = outputs_val[0][0]

            input_image_path = self.val_files[batch_idx]['image']
            image_name = input_image_path.split("/")[3].split(".")[0]
            image_number = image_name.split("_")[0][1:]

            z_mesh_tl_epi = self.ground_truth_latents_epi[image_number]
            z_prediction_epi = self.get_prediction_latent_from_deepssm(predictions_val)

            # Loss functions
            mse_epi_tl = MSE(z_prediction_epi, z_mesh_tl_epi)
            mse_epi_rel_tl_loss = MSE(z_prediction_epi, z_mesh_tl_epi) / MSE(z_prediction_epi * 0, z_mesh_tl_epi)

            l1_epi_tl = torch.mean(torch.abs(z_prediction_epi - z_mesh_tl_epi))
            l1_epi_rel_tl_loss = torch.mean(torch.abs(z_prediction_epi - z_mesh_tl_epi)) / torch.mean(
                torch.abs(z_prediction_epi * 0 - z_mesh_tl_epi))

            z_prediction_epi_normalized = torch.nn.functional.log_softmax(z_prediction_epi, dim=1)
            z_mesh_tl_epi_normalized = torch.nn.functional.softmax(z_mesh_tl_epi, dim=1)
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            kldiv_epi_tl = kl_loss(z_prediction_epi_normalized, z_mesh_tl_epi_normalized)

            # Log loss values
            self.log("val_mse_epi_tl", mse_epi_tl.item(), on_step=False, on_epoch=True)
            self.log("val_mse_epi_rel_tl_loss", mse_epi_rel_tl_loss.item(), on_step=False, on_epoch=True)

            self.log("val_l1_epi_tl", l1_epi_tl.item(), on_step=False, on_epoch=True)
            self.log("val_l1_epi_rel_tl_loss", l1_epi_rel_tl_loss.item(), on_step=False, on_epoch=True)

            self.log("val_kldiv_epi_tl", kldiv_epi_tl.item(), on_step=False, on_epoch=True)

            # Set latent loss
            latent_loss = l1_epi_rel_tl_loss
            self.log("val_latent_loss", latent_loss.item(), on_step=False, on_epoch=True)

        # Add the dice loss and latent loss with a weight
        loss_val = dice_loss + latent_loss

        # Val dice metric
        outputs_val = [self.post_pred(i) for i in decollate_batch(outputs_val)]
        labels_val = [self.post_label(i) for i in decollate_batch(labels_val)]
        self.val_metric_function(y_pred=outputs_val, y=labels_val)

        self.log("val_loss_step", loss_val.item(), on_step=True, on_epoch=False)
        self.val_losses_on_step.append({"val_loss": loss_val})

        return {"val_loss": loss_val}

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

        # Save the prediction as image
        input_image_path = self.test_files[batch_idx]['label']
        image_name = input_image_path.split("/")[3].split(".")[0]
        image_number = image_name.split("_")[0][1:]
        image_data, image_header = nrrd.read(input_image_path)

        predictions_numpy = torch.argmax(outputs_test, dim=1).cpu().numpy().astype(np.uint8)[0]
        output_path = self.predictions_test_path + f"test_img_{image_number}.nrrd"
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
