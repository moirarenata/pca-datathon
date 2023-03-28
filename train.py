# import libraries
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import max, min
from torch.nn import MSELoss
import math
import tifffile
import os
import tempfile
import scipy.io
from glob import glob
import mat73
from PIL import Image
from tifffile import imsave, imwrite, imread
from scipy.io import savemat

from EarlyStopper import EarlyStopper

import monai
from monai.networks.nets import UNet
from monai.data import (
    ArrayDataset,
    create_test_image_2d,
    DataLoader,
    Dataset,
    list_data_collate,
    PILReader,
)
from monai.data.utils import decollate_batch
from monai.apps import CrossValidation
from monai.networks.layers import Norm
from monai.data.utils import partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import MSEMetric
from monai.metrics.regression import SSIMMetric
from monai.losses import DiceLoss
from monai.utils import first, set_determinism
from monai.handlers.utils import from_engine
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirstD,
    LoadImage,
    LoadImageD,
    LoadImaged,
    Orientationd,
    CropForegroundd,
    SpatialCropd,
    ScaleIntensity,
    ScaleIntensityRanged,
    NormalizeIntensity,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
    SaveImaged,
    Invertd,
)
from monai.visualize import plot_2d_or_3d_image
from monai.data import PILWriter

from evaluation_metrics import print_and_log, process_img_saving


def main(tempdir):
    # setup data directory
    '''
    # directory = r'E:\DNN OCT Reconstruction\data'
    directory = '/project/6007991/cydan199/unet_monai/data/'
    if not os.path.exists(directory):
        # os.makedirs(data_root_folder)
        print("Cannot find the folder of training data")

    # read all files in the directory
    raw_data = os.listdir(directory)

    # define the initial volume
    noise_dataset = []  # matrix contains filenames of noisy data
    gt_dataset = []  # matrix contains filenames of ground-truth (clean) data

    for filename in raw_data:
        #        if filename.endswith('Chennel_A.mat') or filename.endswith('Chennel_B.mat')

        # select and record the noisy image data from Channel A
        if filename.endswith('Chennel_A.mat'):
            noise_dataset.append(os.path.join(directory, filename))
        # select and record the ground truth data
        elif filename.endswith('Chennel_A_B.mat'):
            gt_dataset.append(os.path.join(directory, filename))

    noise_dataset = sorted(noise_dataset)
    gt_dataset = sorted(gt_dataset)
    # print(noise_dataset)
    # print(gt_dataset)

    # for code checking, use a single volume data first
    #    check_imset = noise_dataset[0]
    #    chect_gtset = gt_dataset[0]
    #    print(check_imset)
    # im_array = imread(check_imset)
    # gt_array imread(chect_gtset)
    # read image data and save as matrix
    #    im_array = mat73.loadmat(check_imset)['Chennel_A']
    #    gt_array = mat73.loadmat(chect_gtset)['Chennel_A_B']
    # print(im_array.shape)
    # print(gt_array.shape)

    # for code checking, use a single volume data first (20 volumes)
    im_array = mat73.loadmat(noise_dataset[0])['Chennel_A']
    gt_array = mat73.loadmat(gt_dataset[0])['Chennel_A_B']
    #    print(check_imset)

    # read image data and save as matrix
    num_files = len(noise_dataset)
    print(num_files)

    for i in range(1, num_files):
        im_array = numpy.dstack([im_array, mat73.loadmat(noise_dataset[i])['Chennel_A']])
        gt_array = numpy.dstack([gt_array, mat73.loadmat(gt_dataset[i])['Chennel_A_B']])

    print(im_array.shape)
    print(gt_array.shape)

    # temporarily save each frame as a separate file
    num_pairs = im_array.shape[2]

    print(num_pairs)

    # temporarily save each frame as a separate file
    # num_pairs = im_array.shape[2]
    #    tempdir = r'D:\YC-oct-data\tempdir'
    # create a temporary directory and save the 1500 image, mask pairs
    #    print(Image.fromarray(im_array[:, :, 0]))

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(num_pairs):
        #        im_temp = process_img_saving(im_array[:, :, i])
        #        gt_temp = process_img_saving((gt_array[:, :, i]))

        im_temp = Image.fromarray(im_array[:, :, i])
        gt_temp = Image.fromarray(gt_array[:, :, i])

        im_temp.save(os.path.join(tempdir, f"img{i:d}.tiff"))
        gt_temp.save(os.path.join(tempdir, f"gt{i:d}.tiff"))
        # imwrite(os.path.join(tempdir, f"gt{i:d}.tiff"), gt_array[:, :, i].astype())
        # Image.fromarray(gt_array[:, :, i]).save(os.path.join(tempdir, f"gt{i:d}.tiff"))
    '''

    im = sorted(glob(os.path.join(tempdir, "img*.tiff")))
    gt = sorted(glob(os.path.join(tempdir, "gt*.tiff")))

    # match the noisy images with corresponding ground truth and save as zip
    data_files = [
        {"noisy_image": img, "ground_truth": gdtr}
        for img, gdtr in zip(im, gt)
    ]

    # data partition
    data_partition = partition_dataset(
        data=data_files,
        # 0.7 percent training, 0.2 percent validation, 0.1 percent test
        ratios=[0.7, 0.2, 0.1],
        shuffle=False,
    )

    train_files = data_partition[0]
    val_files = data_partition[1]
    # test_files = data_partition[2]

    #   loader = LoadImageD(keys=["noisy_image", "ground_truth"], reader=PILReader)
    #    data_dict = loader(train_files[0])
    #    print(data_dict["noisy_image"].shape)
    # define train transformation
    train_transforms = Compose(
        [
            LoadImageD(keys=["noisy_image", "ground_truth"], reader=PILReader),
            EnsureChannelFirstD(keys=["noisy_image", "ground_truth"]),
            EnsureTyped(keys=["noisy_image", "ground_truth"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["noisy_image", "ground_truth"], reader=PILReader),
            EnsureChannelFirstD(keys=["noisy_image", "ground_truth"]),
            EnsureTyped(keys=["noisy_image", "ground_truth"]),
        ]
    )

    # define dataset, data loader
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4,
                              collate_fn=list_data_collate)  # pin_memory: if true, data loader will copy tensor into CUDA
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["noisy_image"].shape, check_data["ground_truth"].shape)

    # create training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # create validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

    # matrix to indicate validation performance (UNSURE)
    #    ssim_metric = SSIMMetric(spatial_dims=2)

    # Compute average Dice score between two tensors. ** Dice score is commonly used in CNN **
    mse_metric = MSEMetric(reduction='mean', get_not_nans=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32, 64),  # 5 layer network
        strides=(1, 1, 1, 1),
    ).to(device)

    loss_function = MSELoss()  # keep parameters as default
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)  # usually use 1e-5
    # post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # to evaluate the loss function
    # start a typical PyTorch training
    val_interval = 1
    # or float('inf')
    best_metric = 10000
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()

    total_epochs = 10000

    # define the save directory
    # save_path = r'E:\DNN OCT Reconstruction'
    save_path = '/project/6007991/cydan199/unet_monai/results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_save_path = os.path.join(save_path, 'models')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # continue training from previous model
    existed_model_path = os.path.join(model_save_path, 'best_metric_model_octreconstruction_dict.pth')
    if os.path.exists(existed_model_path):
        model.load_state_dict(
            torch.load(existed_model_path, map_location=torch.device('cpu')))

    early_stopper = EarlyStopper(patience=2, min_delta=0)

    for epoch in range(total_epochs):
        print("-" * total_epochs)
        print(f"epoch {epoch + 1}/{total_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1  # increment the epoch number
            inputs, GTs = batch_data["noisy_image"].to(device), batch_data["ground_truth"].to(device)
            print(f"input shape: {inputs.shape}")
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_function(outputs, GTs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        with open(os.path.join(save_path, 'train_performance_logger.txt'), "a") as train_log_file:
            train_log_file = print_and_log(
                '[TRAIN]: epoch, train_loss: |%d|%.3f|' % (epoch, epoch_loss), train_log_file)

        mat_save_path = os.path.join(save_path, 'mats')
        if not os.path.exists(mat_save_path):
            os.makedirs(mat_save_path)

        # define the image saving directory
        tensors_save_path = os.path.join(save_path, 'tensors')
        if not os.path.exists(tensors_save_path):
            os.makedirs(tensors_save_path)

        images_save_path = os.path.join(save_path, 'images')
        if not os.path.exists(images_save_path):
            os.makedirs(images_save_path)

        # validate the model performance
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_grtrs = None
                val_outputs = None
                for frame, val_data in enumerate(val_loader):
                    val_images, val_grtrs = val_data["noisy_image"].to(device), val_data["ground_truth"].to(device)
                    #                    roi_size = (500, 500)
                    #                    sw_batch_size = 4
                    #                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = model(val_images.to(device))
                    # val_outputs = [post_trans(i) for i in decollate_batch(val_images)]
                    # compute metric for current iteration
                    #                    ssim_metric(y_pred=val_outputs, y=val_gtrs)
                    mse_metric(y_pred=val_outputs, y=val_grtrs)
                    # print(f"val_output shape: {val_outputs[0][0, :, :].shape}")
                    for output in range(len(val_outputs)):
                        val_orig = val_images.astype(numpy.float32)
                        val_gt = val_grtrs.astype(numpy.float32)
                        val_predicted = val_outputs[output].astype(numpy.float32)
                        '''
                        # save mat file
                        savemat(os.path.join(mat_save_path,
                                             str(epoch + 1) + "_frame_" + str(frame) + "_origin.mat"),
                                {"Channel_A": val_orig})
                        savemat(os.path.join(mat_save_path,
                                             str(epoch + 1) + "_frame_" + str(frame) + "_gtruth.mat"),
                                {"Channel_A_B": val_gt})
                        savemat(os.path.join(mat_save_path,
                                             str(epoch + 1) + "_frame_" + str(frame) + "_predicted.mat"),
                                {"Pred_Channel_A": val_predicted})
                        # save tensor
                        
                        torch.save(val_orig, os.path.join(tensors_save_path,
                                                          str(epoch + 1) + "_frame_" + str(frame) + "_origin.pt"))
                        torch.save(val_gt, os.path.join(tensors_save_path,
                                                        str(epoch + 1) + "_frame_" + str(frame) + "_gtruth.pt"))
                        torch.save(val_gt, os.path.join(tensors_save_path,
                                                        str(epoch + 1) + "_frame_" + str(frame) + "_gtruth.pt"))
                        '''
                        # save images
                        '''
                        imwrite(
                            os.path.join(images_save_path, str(epoch + 1) + "_frame_" + str(frame) + "_origin.tiff"),
                            val_orig)
                        imwrite(
                            os.path.join(images_save_path, str(epoch + 1) + "_frame_" + str(frame) + "_gtruth.tiff"),
                            val_gt)
                        imwrite(
                            os.path.join(images_save_path, str(epoch + 1) + "_frame_" + str(frame) + "_predicted.tiff"),
                            val_predicted)
                        '''
                    # save mat file (only save the last frame per epoch)
                    savemat(os.path.join(mat_save_path,
                                         str(epoch + 1) + "_frame_" + str(frame) + "_origin.mat"),
                            {"Channel_A": val_orig})
                    savemat(os.path.join(mat_save_path,
                                         str(epoch + 1) + "_frame_" + str(frame) + "_gtruth.mat"),
                            {"Channel_A_B": val_gt})
                    savemat(os.path.join(mat_save_path,
                                         str(epoch + 1) + "_frame_" + str(frame) + "_predicted.mat"),
                            {"Pred_Channel_A": val_predicted})
                # aggregate the final mean square error result
                metric = mse_metric.aggregate().item()
                # reset the status for next validation round
                mse_metric.reset()
                metric_values.append(metric)

                if early_stopper.early_stop(metric, best_metric):
                    # print('true')
                    print(
                        "current epoch: {} current mean square loss: {:.4f} best mean square loss: {:.4f} at epoch {}".format(
                            epoch + 1, metric, best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_mse", metric, epoch + 1)
                    with open(os.path.join(save_path, 'val_performance_logger.txt'), "a") as val_log_file:
                        val_log_file = print_and_log(
                            '[VAL]: epoch, val_loss, best_mse_loss, best_metric_epoch: |%d|%.3f|%.3f|%d|' % (
                                epoch + 1, metric, best_metric, best_metric_epoch), val_log_file)
                    print('Reach the optima and stop training.')
                    break
                else:
                    # print("false")
                    if metric < best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(),
                                   os.path.join(model_save_path, "best_metric_model_octreconstruction_dict.pth"))
                        print("saved new best metric model")

                print(
                    "current epoch: {} current mean square loss: {:.4f} best mean square loss: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mse", metric, epoch + 1)
                with open(os.path.join(save_path, 'val_performance_logger.txt'), "a") as val_log_file:
                    val_log_file = print_and_log(
                        '[VAL]: epoch, val_loss, best_mse_loss, best_metric_epoch: |%d|%.3f|%.3f|%d|' % (
                            epoch + 1, metric, best_metric, best_metric_epoch), val_log_file)
                '''
                if metric < best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(save_path, "best_metric_model_octreconstruction_dict.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean square loss: {:.4f} best mean square loss: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mse", metric, epoch + 1)
                with open(os.path.join(save_path, 'val_performance_logger.txt'), "a") as val_log_file:
                    val_log_file = print_and_log(
                        '[VAL]: epoch, val_loss, best_mse_loss, best_metric_epoch: |%d|%.3f|%.3f|%d|' % (epoch + 1, metric, best_metric, best_metric_epoch), val_log_file)
                '''
    # save validation matrix and epoch loss matrix
    metric_save_path = os.path.join(save_path, 'metrics')
    if not os.path.exists(metric_save_path):
        os.makedirs(metric_save_path)

    savemat(os.path.join(metric_save_path, "epoch_loss_values.mat"), {"epoch_loss_values": epoch_loss_values})
    savemat(os.path.join(metric_save_path, "val_metric_values.mat"), {"val_metric_values": metric_values})

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
#    with tempfile.TemporaryDirectory() as tempdir:
#        print(tempdir)

    # tempdir = os.path.join('/home/cydan199/projects/rrg-msarunic/cydan199/unet_monai', 'tiffdata')
    tempdir = os.path.join(r'D:\YC-oct-data\tempdir')
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    main(tempdir)
















