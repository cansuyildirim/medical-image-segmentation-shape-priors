"""
Ref: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
Ref: https://github.com/SCIInstitute/ShapeWorks/blob/master/
"""
import os
import random
import subprocess
from glob import glob

import nibabel as nib
import pytorch_lightning as pl
import shapeworks as sw
from DeepSSMUtils import model
from DeepSSMUtils.loaders import get_particles
from DeepSSMUtils.losses import MSE
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses.dice import *
from monai.metrics import DiceMetric
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
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

        self.loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)

        self.train_metric_function = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.val_metric_function = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.test_metric_function = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4, amsgrad=True)

        self.post_pred = Compose(
            [EnsureType("tensor", device=torch.device("cuda") if torch.cuda.is_available() else "cpu"),
             AsDiscrete(argmax=True, to_onehot=3)])
        self.post_label = Compose(
            [EnsureType("tensor", device=torch.device("cuda") if torch.cuda.is_available() else "cpu"),
             AsDiscrete(to_onehot=3)])

        self.val_losses_on_step = []
        self.test_losses_on_step = []

        self.best_val_metric = 0
        self.best_val_epoch = 0
        self.best_test_metric = 0

        # Prediction results folders
        self.predictions_dir = os.getcwd() + "/predictions_dir"

        self.endo_predictions_path = self.predictions_dir + "/imp_pred_endo/"
        self.endo_predictions_meshes_path = self.predictions_dir + "/imp_pred_meshes_endo/"

        self.epi_predictions_path = self.predictions_dir + "/imp_pred_epi/"
        self.epi_predictions_meshes_path = self.predictions_dir + "/imp_pred_meshes_epi/"

        self.endo_predictions_path_val = self.predictions_dir + "/val_imp_pred_endo/"
        self.endo_predictions_meshes_path_val = self.predictions_dir + "/val_imp_pred_meshes_endo/"

        self.epi_predictions_path_val = self.predictions_dir + "/val_imp_pred_epi/"
        self.epi_predictions_meshes_path_val = self.predictions_dir + "/val_imp_pred_meshes_epi/"

        check_path_exists(self.predictions_dir)
        check_path_exists(self.endo_predictions_path)
        check_path_exists(self.endo_predictions_meshes_path)
        check_path_exists(self.epi_predictions_path)
        check_path_exists(self.epi_predictions_meshes_path)
        check_path_exists(self.endo_predictions_path_val)
        check_path_exists(self.endo_predictions_meshes_path_val)
        check_path_exists(self.epi_predictions_path_val)
        check_path_exists(self.epi_predictions_meshes_path_val)

        check_path_exists(os.getcwd() + "/predictions_test")

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_image_latent_from_tldeepssm(input_image_path, layer):
        """
        This method finds the latent code of the groomed input image using TL-DeepSSM Image Encoder
        :param input_image_path: groomed input image path
        :param layer: epi or endo
        :return: latent code of groomed input image
        """
        image = sw.Image(input_image_path).toArray(copy=True, for_viewing=True)

        img = torch.FloatTensor(np.array(image))
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)

        path = os.getcwd() + "/Output/deep_ssm_mitea"
        if layer == "epi":
            network_path = path + "/tl_epi"
        else:
            network_path = path + "/tl_endo"

        # Initialize DeepSSMNet_TLNet with config
        tl_config_path = network_path + "/mitea_deepssm.json"
        tl_net = model.DeepSSMNet_TLNet(tl_config_path)

        # Load the best trained model
        model_dir = network_path + "/mitea_deepssm/"
        best_model_path = os.path.join(model_dir, 'best_model.torch')
        tl_net.load_state_dict(torch.load(best_model_path))

        # Set the model to evaluation mode
        tl_net = tl_net.to(device)
        tl_net.eval()

        # Freeze the encoder
        for param in tl_net.ImageEncoder.parameters():
            param.requires_grad = False

        # Get latent code of groomed input image
        zt, _ = tl_net.ImageEncoder(img)

        return zt

    def prepare_data(self):
        # Create dictionaries for train, val and test files
        train_files_images = sorted(glob(os.path.join("data/train/", "images", "*.nii")))
        train_files_labels = sorted(glob(os.path.join("data/train/", "labels", "*.nii")))
        self.train_files = [{"image": image, "label": label} for image, label in
                            zip(train_files_images, train_files_labels)]

        val_files_images = sorted(glob(os.path.join("data/val/", "images", "*.nii")))
        val_files_labels = sorted(glob(os.path.join("data/val/", "labels", "*.nii")))
        self.val_files = [{"image": image, "label": label} for image, label in
                          zip(val_files_images, val_files_labels)]

        test_files_images = sorted(glob(os.path.join("data/test/", "images", "*.nii")))
        test_files_labels = sorted(glob(os.path.join("data/test/", "labels", "*.nii")))
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
        # using TLDeepSSM frozen image encoder
        self.ground_truth_latents_tl_epi = {}
        self.ground_truth_latents_tl_endo = {}

        groomed_images_folders = {"epi": os.getcwd() + "/Output/deep_ssm_mitea/tl_epi/data/groomed_images/",
                                  "endo": os.getcwd() + "/Output/deep_ssm_mitea/tl_endo/data/groomed_images/"}

        # Use groomed images since they go into image encoder
        for layer, groomed_images_folder in groomed_images_folders.items():
            groomed_images = [i for i in os.listdir(groomed_images_folder) if i.endswith(".nrrd")]

            for groomed_image in groomed_images:
                groomed_image_path = os.path.join(groomed_images_folder, groomed_image)
                groomed_image_number = groomed_image.split("_")[0][1:]  # e.g. 001

                gt_latent_tl = self.get_image_latent_from_tldeepssm(groomed_image_path, layer)  # from TLDeepSSM

                if layer == "epi":
                    self.ground_truth_latents_tl_epi[groomed_image_number] = gt_latent_tl
                else:
                    self.ground_truth_latents_tl_endo[groomed_image_number] = gt_latent_tl

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

    def get_mesh_particles(self, mesh_path, number, layer):
        """
        This method finds the correspondence points (particles) of the mesh based on ShapeWorks particle-based modeling
        approach
        :param mesh_path: predicted mesh path
        :param number: image number (e.g. 001)
        :param layer: endo or epi
        :return: correspondence points of the predicted mesh
        """
        name = f'pred_{layer}_img{number}_{self.current_epoch}'

        path = os.getcwd() + "/Output/deep_ssm_mitea"
        if layer == "epi":
            network_path = path + "/tl_epi"
            data_dir = os.getcwd() + "/mesh_particles_epi/"
        else:
            network_path = path + "/tl_endo"
            data_dir = os.getcwd() + "/mesh_particles_endo/"

        check_path_exists(data_dir)
        check_path_exists(data_dir + name + "_particles/")

        # Get meanshape in world particles
        meanshape_world_particles_path = network_path + "/data/meanshape_world.particles"
        mean_shape = np.loadtxt(meanshape_world_particles_path)

        subjects = []
        project_location = data_dir

        subject = sw.Subject()
        subject.set_number_of_domains(1)
        rel_mesh_files = sw.utils.get_relative_paths([mesh_path], project_location)
        rel_groom_files = sw.utils.get_relative_paths([mesh_path], project_location)
        initial_particles = sw.utils.transformParticles(mean_shape, np.eye(4), inverse=True)
        initial_particle_file = data_dir + name + "_particles/" + name + "_local.particles"
        np.savetxt(initial_particle_file, initial_particles)
        rel_particle_files = sw.utils.get_relative_paths([initial_particle_file], project_location)
        subject.set_original_filenames(rel_mesh_files)
        subject.set_groomed_filenames(rel_groom_files)
        subject.set_landmarks_filenames(rel_particle_files)
        subject.set_extra_values({"fixed": "no"})
        subjects.append(subject)

        project = sw.Project()
        project.set_subjects(subjects)
        parameters = sw.Parameters()

        parameter_dictionary = {
            "number_of_particles": 128,
            "use_normals": 0,
            "normals_strength": 10.0,
            "iterations_per_split": 1500,
            "optimization_iterations": 1000,
            "starting_regularization": 100,
            "ending_regularization": 0.1,
            "relative_weighting": 10,
            "initial_relative_weighting": 0.1,
            "procrustes": 0,
            "procrustes_interval": 0,
            "procrustes_scaling": 0,
            "save_init_splits": 1,
            "multiscale": 0,
            "multiscale_particles": 32,
            "use_landmarks": 1,
            "use_fixed_subjects": 1,
            "narrow_band": 1e10,
            "fixed_subjects_column": "fixed",
            "fixed_subjects_choice": "yes",
        }

        for key in parameter_dictionary:
            parameters.set(key, sw.Variant(parameter_dictionary[key]))
        project.set_parameters("optimize", parameters)
        # Set studio parameters
        studio_dictionary = {
            "show_landmarks": 0,
            "tool_state": "analysis"
        }
        studio_parameters = sw.Parameters()
        for key in studio_dictionary:
            studio_parameters.set(key, sw.Variant(studio_dictionary[key]))
        project.set_parameters("studio", studio_parameters)
        spreadsheet_file = data_dir + name + ".xlsx"
        project.save(spreadsheet_file)

        # Run optimization
        optimize_cmd = ('shapeworks optimize --progress --name ' + spreadsheet_file).split()
        subprocess.check_call(optimize_cmd)

        project = sw.Project()
        project.load(spreadsheet_file)

        # Get world particle files
        world_particles = data_dir + name + "_particles/" + name + "_world.particles"
        return world_particles

    def get_prediction_latent(self, path, number, layer):
        """
        This method finds the latent code of the predicted mesh using trained and frozen TL-DeepSSM Correspondence
        Encoder

        :param path: path to predicted mesh
        :param number: image number (e.g. 001)
        :param layer: endo or epi
        :return: latent code of the predicted mesh
        """
        # Get particles of the predicted mesh using ShapeWorks particle-based shape modeling method
        mesh_particles = self.get_mesh_particles(path, number, layer)

        path = os.getcwd() + "/Output/deep_ssm_mitea"
        if layer == "epi":
            network_path = path + "/tl_epi"
        else:
            network_path = path + "/tl_endo"

        # Initialize DeepSSMNet_TLNet with config
        tl_config_path = network_path + "/mitea_deepssm.json"
        tl_net = model.DeepSSMNet_TLNet(tl_config_path)

        # Load the best trained model
        model_dir = network_path + "/mitea_deepssm/"
        best_model_path = os.path.join(model_dir, 'best_model.torch')
        tl_net.load_state_dict(torch.load(best_model_path))

        # Set the model to evaluation mode
        tl_net = tl_net.to(device)
        tl_net.eval()

        # Freeze the encoder
        for param in tl_net.CorrespondenceEncoder.parameters():
            param.requires_grad = False

        # Prepare the particles for the encoder
        mdl = get_particles(mesh_particles)
        mdl_target = torch.FloatTensor(np.array(mdl))
        mdl_target = torch.unsqueeze(mdl_target, 0)
        mdl_target = mdl_target.to(device)
        pt1 = mdl_target.view(-1, mdl_target.shape[1] * mdl_target.shape[2])

        # Get latent code of predicted mesh
        latent_code = tl_net.CorrespondenceEncoder(pt1)

        return latent_code

    def get_predicted_mesh(self, predictions, input_image_path, predictions_path, predictions_meshes_path, pred_layer):
        """
        This function creates the prediction mesh and align it based on the reference mesh in DeepSSM

        :param predictions: prediction mask of endocardium or epicardium
        :param input_image_path: input image from train_files
        :param predictions_path: saved prediction images path
        :param predictions_meshes_path: saved prediction meshes path
        :param pred_layer: prediction layer, endo or epi
        :return: image number and predicted mesh path
        """
        image_data = nib.load(input_image_path)

        image_name = input_image_path.split("/")[3].split(".")[0]
        image_number = image_name.split("_")[1]
        image_affine = image_data.affine
        image_header = image_data.header

        # Save the prediction as nifti image
        nifti_filename = f"pred_{pred_layer}_img{image_number}_{self.current_epoch}.nii"
        nifti_filepath = os.path.join(predictions_path, nifti_filename)

        nifti_image = nib.Nifti1Image(predictions, affine=image_affine, header=image_header)
        nib.save(nifti_image, nifti_filepath)

        predicted_nifti_path = predictions_path + nifti_filename

        # Convert to mesh
        predicted_mesh_path = predictions_meshes_path + f"pred_{pred_layer}_img{image_number}_{self.current_epoch}.vtk"

        img = sw.Image(predicted_nifti_path)
        img.antialias(1)
        img.gaussianBlur(2)  # makes smooth
        mesh = img.toMesh(1)  # convert image to mesh

        # Align the mesh based on reference mesh
        path = os.getcwd() + "/Output/deep_ssm_mitea"
        if pred_layer == "epi":
            data_path = path + "/tl_epi/data"
        else:
            data_path = path + "/tl_endo/data"

        ref_mesh_path = data_path + "/reference.vtk"
        ref_mesh = sw.Mesh(ref_mesh_path)

        rigid_transform = mesh.createTransform(ref_mesh, sw.Mesh.AlignmentType.Rigid, 100)
        mesh.applyTransform(rigid_transform)

        # Write out the mesh
        mesh.write(predicted_mesh_path)

        return image_number, predicted_mesh_path

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"].to(device), batch["label"].to(device)

        # Get the dice loss
        output = self.forward(images)
        dice_loss = self.loss_function(output, labels)
        self.log("train_dice_loss", dice_loss.item(), on_step=False, on_epoch=True)

        latent_loss = 0
        # Get latent loss after epoch 20
        if self.current_epoch >= 20:
            # Get prediction mask
            predictions = torch.argmax(output, dim=1).cpu().numpy().astype(np.uint8)
            predictions = predictions[0]

            # Input image path (to get affine and header of the original image)
            input_image_path = self.train_files[batch_idx]['image']

            # Endocardium (inner) predictions
            image_number, predicted_endo_mesh_path = self.get_predicted_mesh(predictions, input_image_path,
                                                                             self.endo_predictions_path,
                                                                             self.endo_predictions_meshes_path, "endo")

            # Epicardium (outer) predictions
            epicardium_predictions = predictions.copy()
            epicardium_predictions[epicardium_predictions == 2] = 1

            image_number, predicted_epi_mesh_path = self.get_predicted_mesh(epicardium_predictions, input_image_path,
                                                                            self.epi_predictions_path,
                                                                            self.epi_predictions_meshes_path, "epi")

            # Get latent codes of prediction and ground truth and log the losses
            # ENDO
            z_prediction_endo = self.get_prediction_latent(path=predicted_endo_mesh_path, number=image_number, layer="endo")
            z_image_tl_endo = self.ground_truth_latents_tl_endo[image_number]

            mse_endo_tl = MSE(z_prediction_endo, z_image_tl_endo)
            mse_endo_tl_rel = MSE(z_prediction_endo, z_image_tl_endo) / MSE(z_prediction_endo * 0, z_image_tl_endo)
            l1_endo_tl = torch.mean(torch.abs(z_prediction_endo - z_image_tl_endo))
            l1_endo_tl_rel = torch.mean(torch.abs(z_prediction_endo - z_image_tl_endo)) / torch.mean(torch.abs(z_prediction_endo * 0 - z_image_tl_endo))

            self.log("train_mse_endo_tl", mse_endo_tl.item(), on_step=False, on_epoch=True)
            self.log("train_l1_endo_tl", l1_endo_tl.item(), on_step=False, on_epoch=True)

            self.log("train_mse_endo_tl_rel", mse_endo_tl_rel.item(), on_step=False, on_epoch=True)
            self.log("train_l1_endo_tl_rel", l1_endo_tl_rel.item(), on_step=False, on_epoch=True)

            # EPI
            z_prediction_epi = self.get_prediction_latent(path=predicted_epi_mesh_path, number=image_number, layer="epi")
            z_image_tl_epi = self.ground_truth_latents_tl_epi[image_number]

            mse_epi_tl = MSE(z_prediction_epi, z_image_tl_epi)
            mse_epi_tl_rel = MSE(z_prediction_epi, z_image_tl_epi) / MSE(z_prediction_epi * 0, z_image_tl_epi)
            l1_epi_tl = torch.mean(torch.abs(z_prediction_epi - z_image_tl_epi))
            l1_epi_tl_rel = torch.mean(torch.abs(z_prediction_epi - z_image_tl_epi)) / torch.mean(torch.abs(z_prediction_epi * 0 - z_image_tl_epi))

            self.log("train_mse_epi_tl", mse_epi_tl.item(), on_step=False, on_epoch=True)
            self.log("train_l1_epi_tl", l1_epi_tl.item(), on_step=False, on_epoch=True)

            self.log("train_mse_epi_tl_rel", mse_epi_tl_rel.item(), on_step=False, on_epoch=True)
            self.log("train_l1_epi_tl_rel", l1_epi_tl_rel.item(), on_step=False, on_epoch=True)

            latent_loss = l1_endo_tl_rel + l1_epi_tl_rel
            self.log("latent_loss", latent_loss.item(), on_step=False, on_epoch=True)

        # Add the dice loss and latent loss with a weight
        loss = dice_loss + latent_loss
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True)

        # Train dice metric
        post_processed_output = [self.post_pred(pred) for pred in decollate_batch(output)]
        post_processed_labels = [self.post_label(label) for label in decollate_batch(labels)]

        self.train_metric_function(y_pred=post_processed_output, y=post_processed_labels)

        return {"loss": loss}

    def on_train_epoch_end(self):
        train_metric_epoch = self.train_metric_function.aggregate().item()
        self.train_metric_function.reset()

        self.log("train_dice_metric_epoch", train_metric_epoch, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        images_val, labels_val = batch["image"].to(device), batch["label"].to(device)

        # Get the dice loss
        outputs_val = self.forward(images_val)
        dice_loss = self.loss_function(outputs_val, labels_val)
        self.log("val_dice_loss", dice_loss.item(), on_step=False, on_epoch=True)

        latent_loss = 0
        # Get latent loss after epoch 15
        if self.current_epoch >= 20:
            # Get prediction mask
            predictions = torch.argmax(outputs_val, dim=1).cpu().numpy().astype(np.uint8)
            predictions = predictions[0]

            # Get original image affine and header
            input_image_path = self.val_files[batch_idx]['image']

            # Endocardium (inner) predictions
            image_number, predicted_endo_mesh_path = self.get_predicted_mesh(predictions, input_image_path,
                                                                             self.endo_predictions_path_val,
                                                                             self.endo_predictions_meshes_path_val, "endo")

            # Epicardium (outer) predictions
            epicardium_predictions = predictions.copy()
            epicardium_predictions[epicardium_predictions == 2] = 1
            image_number, predicted_epi_mesh_path = self.get_predicted_mesh(epicardium_predictions, input_image_path,
                                                                            self.epi_predictions_path_val,
                                                                            self.epi_predictions_meshes_path_val, "epi")

            # Get latent codes of prediction and ground truth
            # ENDO
            z_prediction_endo = self.get_prediction_latent(path=predicted_endo_mesh_path, number=image_number,
                                                           layer="endo")
            z_image_tl_endo = self.ground_truth_latents_tl_endo[image_number]

            mse_endo_tl = MSE(z_prediction_endo, z_image_tl_endo)
            mse_endo_tl_rel = MSE(z_prediction_endo, z_image_tl_endo) / MSE(z_prediction_endo * 0, z_image_tl_endo)
            l1_endo_tl = torch.mean(torch.abs(z_prediction_endo - z_image_tl_endo))
            l1_endo_tl_rel = torch.mean(torch.abs(z_prediction_endo - z_image_tl_endo)) / torch.mean(
                torch.abs(z_prediction_endo * 0 - z_image_tl_endo))

            self.log("val_mse_endo_tl", mse_endo_tl.item(), on_step=False, on_epoch=True)
            self.log("val_l1_endo_tl", l1_endo_tl.item(), on_step=False, on_epoch=True)

            self.log("val_mse_endo_tl_rel", mse_endo_tl_rel.item(), on_step=False, on_epoch=True)
            self.log("val_l1_endo_tl_rel", l1_endo_tl_rel.item(), on_step=False, on_epoch=True)

            # EPI
            z_prediction_epi = self.get_prediction_latent(path=predicted_epi_mesh_path, number=image_number,
                                                          layer="epi")
            z_image_tl_epi = self.ground_truth_latents_tl_epi[image_number]

            mse_epi_tl = MSE(z_prediction_epi, z_image_tl_epi)
            mse_epi_tl_rel = MSE(z_prediction_epi, z_image_tl_epi) / MSE(z_prediction_epi * 0, z_image_tl_epi)
            l1_epi_tl = torch.mean(torch.abs(z_prediction_epi - z_image_tl_epi))
            l1_epi_tl_rel = torch.mean(torch.abs(z_prediction_epi - z_image_tl_epi)) / torch.mean(
                torch.abs(z_prediction_epi * 0 - z_image_tl_epi))

            self.log("val_mse_epi_tl", mse_epi_tl.item(), on_step=False, on_epoch=True)
            self.log("val_l1_epi_tl", l1_epi_tl.item(), on_step=False, on_epoch=True)

            self.log("val_mse_epi_tl_rel", mse_epi_tl_rel.item(), on_step=False, on_epoch=True)
            self.log("val_l1_epi_tl_rel", l1_epi_tl_rel.item(), on_step=False, on_epoch=True)

            latent_loss = l1_endo_tl_rel + l1_epi_tl_rel
            self.log("val_latent_loss", latent_loss.item(), on_step=False, on_epoch=True)

        # Add the dice loss and latent loss with a weight
        loss = dice_loss + latent_loss

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

        outputs_test = self.forward(images_test)

        # Save the prediction as nifti image
        test_predictions = torch.argmax(outputs_test, dim=1).cpu().numpy().astype(np.uint8)
        test_predictions = test_predictions[0]

        input_image_path = self.test_files[batch_idx]['image']
        image_data = nib.load(input_image_path)
        image_name = input_image_path.split("/")[3].split(".")[0]
        image_number = image_name.split("_")[1]
        image_affine = image_data.affine
        image_header = image_data.header

        # Save the prediction as nifti image
        test_pred_path = os.getcwd() + "/predictions_test"
        nifti = f"test_pred_img{image_number}.nii"
        nifti_filepath = os.path.join(test_pred_path, nifti)
        nifti_image = nib.Nifti1Image(test_predictions, affine=image_affine, header=image_header)
        nib.save(nifti_image, nifti_filepath)

        loss = self.loss_function(outputs_test, labels_test)

        outputs_test = [self.post_pred(i) for i in decollate_batch(outputs_test)]
        labels_test = [self.post_label(i) for i in decollate_batch(labels_test)]

        self.test_metric_function(y_pred=outputs_test, y=labels_test)

        self.log("test_loss_step", loss.item(), on_step=True, on_epoch=False)

        self.test_losses_on_step.append({"test_loss": loss})

        return {"test_loss": loss}

    def on_test_epoch_end(self):
        test_loss_epoch = torch.stack([x["test_loss"] for x in self.test_losses_on_step]).mean()

        test_metric_epoch = self.test_metric_function.aggregate().item()
        self.test_metric_function.reset()

        self.log("test_loss_epoch", test_loss_epoch, on_step=False, on_epoch=True)
        self.log("test_metric_epoch", test_metric_epoch, on_step=False, on_epoch=True)

        if test_metric_epoch > self.best_test_metric:
            self.best_test_metric = test_metric_epoch

        print(f"Best test metric: {self.best_test_metric}")

        self.test_losses_on_step.clear()
