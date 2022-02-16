# main imports
import numpy as np
import argparse
import os
import random

# image processing
from PIL import Image

# deep learning imports
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

# vizualisation
import torchvision.utils as vutils
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# models imports
from models.autoencoders.ushaped import UShapedAutoencoder as AutoEncoder
from models.discriminators.discriminator_v1 import Discriminator

# losses imports
from losses.utils import Loss

from scipy.ndimage import gaussian_filter
from ipfml.processing.transform import get_LAB_L
from ipfml.processing import reconstruction
import cv2

import processing.config as cfg
# logger import
import gym
log = gym.logger
log.set_level(gym.logger.INFO)

# other parameters
NB_IMAGES = 64

BACKUP_MODEL_NAME = "{}_model.pt"
BACKUP_FOLDER = "saved_models"
BACKUP_EVERY_ITER = 1

LEARNING_RATE = 0.0002
REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 20

from predictions.seq_processing.model_choice import prepare_model, choices_input

def get_batch_data(data):

    # reference batch data
    batch_ref_data, _  = data[0]
    batch_input_data, _  = data[1]
    batch_weights, _  = data[2]
    batch_labels, _  = data[3]

    return batch_ref_data, batch_input_data, batch_weights, batch_labels


# initialize weights function
def initialize_weights(arg_class):
  class_name = arg_class.__class__.__name__
  if class_name.find('Conv') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 0.0, 0.02)
  elif class_name.find('BatchNorm') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(arg_class.bias.data, 0)


# Concatenate features and reference data
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class CustomNormalize(object):
    """Normalize image input between 0 and 1.
    """

    def __call__(self, sample):
        
        image = sample.numpy()

        # normalise data
        image = image / np.max(image)

        return torch.from_numpy(image)

def numpy_loader(sample_path):

    image_array = np.load(sample_path)

    # convert as float 32
    image_array = np.array(image_array, 'float32')

    # if gray scale image
    if image_array.ndim < 3:
        image_array = np.expand_dims(image_array, axis=2)

    return image_array

def weights_loader(sample_path):

    image_array = np.load(sample_path)

    # convert as float 32
    return np.array(image_array, 'float32').reshape((1, 1))

def labels_loader(sample_path):

    image_array = np.load(sample_path)

    # convert as float32
    return np.array(image_array, 'float32').reshape((1, 1))

def main():

    load_model = False
    restart = False
    start_epoch = 0
    start_iteration = 0

    parser = argparse.ArgumentParser(description="Generate model using specific reversed GAN procedure")

    parser.add_argument('--folder', type=str, help="folder with train/test folders within all features sub folders")
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=32)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=100)
    parser.add_argument('--weighted', type=int, help='Enable to use weights of current sample when using loss', default=0)
    parser.add_argument('--sequence', type=int, help='expected sequence size', required=True)
    parser.add_argument('--seqlayer', type=str, help='sequence layer (exemple "1,1,1")', required=True)
    parser.add_argument('--choice', type=str, help='input data format', default=choices_input[0], choices=choices_input)
    parser.add_argument('--save', type=str, help='save folder for backup model', default='')
    parser.add_argument('--load', type=str, help='folder backup model', default='')

    args = parser.parse_args()

    p_folder              = args.folder
    p_batch_size          = args.batch_size
    p_epochs              = args.epochs
    p_weighted            = bool(args.weighted)
    p_sequence            = args.sequence
    p_choice              = args.choice
    p_save                = args.save
    p_load                = args.load
    p_seqlayer            = [ int(e) for e in args.seqlayer.split(',') ]

    if len(p_load) > 0:
        load_model = True

    # build data path
    train_path = os.path.join(p_folder, 'train')
    test_path = os.path.join(p_folder, 'test')

    references_train_path = os.path.join(train_path, cfg.references_folder)
    input_train_path = os.path.join(train_path, 'input')
    weights_train_path = os.path.join(train_path, 'weights')
    labels_train_path = os.path.join(train_path, 'labels')

    references_test_path = os.path.join(test_path, cfg.references_folder)
    input_test_path = os.path.join(test_path, 'input')
    weights_test_path = os.path.join(test_path, 'weights')
    labels_test_path = os.path.join(test_path, 'labels')

    print('Data to predict:\n-', references_train_path)
    is_valid_file_f = lambda x: True if str(x).endswith('.npy') else False

    # set references as first params
    img_ref_folder = torchvision.datasets.ImageFolder(references_train_path, loader=numpy_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    img_input_folder = torchvision.datasets.ImageFolder(input_train_path, loader=numpy_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    img_weights_folder = torchvision.datasets.DatasetFolder(weights_train_path, loader=weights_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    img_labels_folder = torchvision.datasets.DatasetFolder(labels_train_path, loader=labels_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))


    image_folders_data = [img_ref_folder, img_input_folder, img_weights_folder, img_labels_folder]

    # shuffle data loader and made possible to keep track well of reference
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(image_folders_data),
        batch_size=p_batch_size, shuffle=True,
        num_workers=0, pin_memory=False)


    test_img_ref_folder = torchvision.datasets.ImageFolder(references_test_path, loader=numpy_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    test_img_input_folder = torchvision.datasets.ImageFolder(input_test_path, loader=numpy_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    test_img_weights_folder = torchvision.datasets.DatasetFolder(weights_test_path, loader=weights_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    test_img_labels_folder = torchvision.datasets.DatasetFolder(labels_test_path, loader=labels_loader, is_valid_file=is_valid_file_f, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    image_folders_data = [img_ref_folder, img_input_folder, img_weights_folder, img_labels_folder]
    test_image_folders_data = [test_img_ref_folder, test_img_input_folder, test_img_weights_folder, test_img_labels_folder]

    # shuffle data loader and made possible to keep track well of reference
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(image_folders_data),
        batch_size=p_batch_size, shuffle=True,
        num_workers=0, pin_memory=False)

    # no use of test dataset for the moment
    # test_loader = torch.utils.data.DataLoader(
    #     ConcatDataset(test_image_folders_data),
    #     batch_size=p_batch_size, shuffle=True,
    #     num_workers=0, pin_memory=False)
             
    train_dataset_batch_size = len(train_loader)

    # creating and loading model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    ##################################
    # Autoencoder model declaration  #
    ##################################
    
    # getting input image size
    for _, data in enumerate(train_loader):
        batch_ref_data, batch_input, _, _ = get_batch_data(data)
        input_n_channels = list(batch_input.size())[1]
        input_img_size = list(batch_input.size())[2]
        output_n_channels = list(batch_input.size())[1]
        output_img_size = list(batch_input.size())[2]
        break

    # define models and loss functions
    # default 3 channels
    autoencoder_ref, autoencoder_mask, discriminator = prepare_model(p_choice, p_seqlayer, output_img_size, p_sequence)

    autoencoder_ref.to(device)
    autoencoder_mask.to(device)
    discriminator.to(device)

    print(autoencoder_ref)
    print(autoencoder_mask)

    # set autoencoder parameters
    autoencoder_ref_parameters = autoencoder_ref.params()
    autoencoder_mask_parameters = autoencoder_mask.params()
    # autoencoder_loss_func = instanciate(p_autoencoder_loss)

    autoencoder_ref_loss_func = Loss('ssim') # default use of SSIM loss
    autoencoder_ref_optimizer = torch.optim.Adam(autoencoder_ref_parameters, lr=LEARNING_RATE, betas=(0.5, 0.999))
    autoencoder_mask_optimizer = torch.optim.Adam(autoencoder_mask_parameters, lr=LEARNING_RATE, betas=(0.5, 0.999))

    print('Autoencoder ref total parameters : ', sum(p.numel() for p in autoencoder_ref_parameters))
    print('Autoencoder mask total parameters : ', sum(p.numel() for p in autoencoder_mask_parameters))

    ####################################
    # Discriminator model declaration  #
    ####################################
    # discriminator = Discriminator(sum(p_seqlayer), output_img_size).to(device)
    discriminator.apply(initialize_weights)

    # discriminator_loss_func = Loss(p_discriminator_loss)
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    print(discriminator)
    print('discriminator total parameters : ', sum(p.numel() for p in discriminator.parameters()))

    print('--------------------------------------------------------')
    print("Train data loader size : ", train_dataset_batch_size)

    # default params
    iteration = 0
    autoencoder_ref_losses = []
    autoencoder_mask_losses = []
    discriminator_losses = []

    # load models checkpoint if exists
    if load_model:

        load_models_folder_path = os.path.join(BACKUP_FOLDER, p_load)

        if not os.path.exists(load_models_folder_path):
            os.makedirs(load_models_folder_path)

        load_global_model_path = None

        if len(os.listdir(load_models_folder_path)) > 0:
            last_epoch_model = sorted(os.listdir(load_models_folder_path))[-1]
            load_global_model_path = os.path.join(last_epoch_model, BACKUP_MODEL_NAME.format('global'))

        if load_global_model_path is None or not os.path.exists(load_global_model_path):
            print('-------------------------')
            print('Model backup not found...')
            print('-------------------------')
        else:

            load_discriminator_model_path = os.path.join(last_epoch_model, BACKUP_MODEL_NAME.format('discriminator'))
            load_autoencoder_ref_model_path = os.path.join(last_epoch_model, BACKUP_MODEL_NAME.format('autoencoder_ref'))
            load_autoencoder_mask_model_path = os.path.join(last_epoch_model, BACKUP_MODEL_NAME.format('autoencoder_mask'))

            # load autoencoder state
            autoencoder_ref_checkpoint = torch.load(load_autoencoder_ref_model_path)
            autoencoder_mask_checkpoint = torch.load(load_autoencoder_mask_model_path)

            autoencoder_ref.load_state_dict(autoencoder_ref_checkpoint['autoencoder_ref_state_dict'])
            autoencoder_ref_optimizer.load_state_dict(autoencoder_ref_checkpoint['optimizer_ref_state_dict'])
            autoencoder_ref_losses = autoencoder_ref_checkpoint['autoencoder_ref_losses']

            autoencoder_mask.load_state_dict(autoencoder_mask_checkpoint['autoencoder_mask_state_dict'])
            autoencoder_mask_optimizer.load_state_dict(autoencoder_mask_checkpoint['optimizer_mask_state_dict'])
            autoencoder_mask_losses = autoencoder_mask_checkpoint['autoencoder_mask_losses']

            autoencoder_ref.train()
            autoencoder_mask.train()

            # load discriminator state
            if os.path.exists(load_discriminator_model_path):
                discriminator_checkpoint = torch.load(load_discriminator_model_path)

                discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
                discriminator_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
                discriminator_losses = discriminator_checkpoint['discriminator_losses']

                discriminator.train()

            # load global state
            global_checkpoint = torch.load(load_global_model_path)

            backup_iteration = global_checkpoint['iteration']
            backup_epochs = global_checkpoint['epochs'] 

            # update context variables
            start_iteration = backup_iteration
            start_epoch = backup_epochs
            restart = True

            print('---------------------------')
            print('Model backup found....')
            print('Restart from epoch', start_epoch)
            print('Restart from iteration', start_iteration)
            print('---------------------------')
        
    # define writer
    writer = SummaryWriter(os.path.join('runs_gans_v2', p_save))

    for epoch in range(p_epochs):

        print('Start epoch:', epoch)
            
        # initialize correct detected from discriminator
        correct_detected = 0
        roc_auc_score_sum = 0

         # check dataset in order to restart
        if train_dataset_batch_size * (epoch + 1) < start_iteration and restart:
            iteration += train_dataset_batch_size
            continue

        # if needed to restart, then restart from expected train_loader element
        if restart:
            nb_viewed_elements = start_iteration % train_dataset_batch_size
            indices = [ i + nb_viewed_elements for i in range(nb_viewed_elements) ]
            
            train_dataset = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_loader.dataset, indices),
                batch_size=p_batch_size, shuffle=True,
                num_workers=0, pin_memory=False)

            print('Restart using the last', len(train_dataset), 'elements of train dataset')
            restart = False
            start_iteration = 0
        else:
            train_dataset = train_loader


        for batch_id, data in enumerate(train_dataset):
            
            if start_iteration > iteration:
                iteration += 1
                continue
            
            # 1. get input batchs and reference

            # get reference and inputs batch data
            batch_ref_data, batch_input, batch_weights, batch_labels = get_batch_data(data)

            # convert batch to specific device
            batch_ref_data = batch_ref_data.to(device)
            batch_input = batch_input.to(device)
            batch_weights = batch_weights.to(device)
            batch_labels = batch_labels.to(device)

            b_size = list(batch_input.size())[0]

            reshaped_weights = batch_weights.reshape(b_size).to(device)
            reshaped_labels= batch_labels.reshape(b_size).to(device)

            # 2.1. Train autoencoder ref..
            # Autoencoder Reference has to produce same input images (Encode => Decode)
                        
            reference_sequence_data = []

            # train over multiple noise level
            for i in range(p_sequence):

                autoencoder_ref_optimizer.zero_grad() 
                
                # print(f'Iteration {i} => {torch.cuda.memory_summary()}')
                predicted_reference = autoencoder_ref(batch_input.data[:, i*3:(i*3)+3])

                autoencoder_ref_loss = autoencoder_ref_loss_func.compute(predicted_reference, batch_ref_data)
            
                reference_sequence_data.append(predicted_reference)

                autoencoder_ref_losses.append(autoencoder_ref_loss.item())

                autoencoder_ref_loss.backward()
                autoencoder_ref_optimizer.step()
            # autoencoder_ref_optimizer.zero_grad() 

            torch_reference_sequence = torch.cat(reference_sequence_data, dim=1)

            # print('Reference sequence', reference_sequence_data.size())
            # 2.2. Train autoencoder mask generator..
            autoencoder_mask_optimizer.zero_grad()

            # no need to pass gradient for these variables
            ref_mask = autoencoder_mask(Variable(torch_reference_sequence, requires_grad=False))
            noisy_mask = autoencoder_mask(Variable(batch_input, requires_grad=False))

            inputs_mask = torch.cat((ref_mask, noisy_mask), dim=1)

            # print('Input mask', inputs_mask.size())

            # 3. train discriminator
            # only if necessary (generator trained well before) - assumption: avoid of local optima                 
            discriminator_optimizer.zero_grad()

            # Pass generated inputs_mask
            discriminator_output = discriminator(inputs_mask)

            # specific use of weighted samples or not
            if p_weighted:
                discriminator_loss_func = torch.nn.BCELoss(reshaped_weights)
            else:
                discriminator_loss_func = torch.nn.BCELoss()

            discriminator_loss = discriminator_loss_func(discriminator_output, reshaped_labels)

            discriminator_losses.append(discriminator_loss.item())

            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            # 3.1. Compute again discrimintator output for autoencoder mask loss

            # combination of expected prediction (noisy / no noisy) and expected reference image
            discriminator_output = discriminator(inputs_mask)

            mask_autoencoder_loss = discriminator_loss_func(discriminator_output, reshaped_labels)

            autoencoder_mask_losses.append(mask_autoencoder_loss.item())
            mask_autoencoder_loss.backward()
            autoencoder_mask_optimizer.step()
            
            # 4. compute accuracy from the epoch
            discriminator_output_label = (discriminator_output > 0.5).float() 

            correct_detected += (discriminator_output_label == reshaped_labels).float().sum()
            
            try:
                roc_auc_score_sum += roc_auc_score(reshaped_labels.detach().cpu().numpy(), discriminator_output.detach().cpu().numpy())
            except:
                print('Not possible to compute ROC score for this batch... Perhaps a lot of precision...')

            discriminator_accuracy = correct_detected / float(((batch_id + 1) * p_batch_size))
            discriminator_auc_roc = roc_auc_score_sum / float(batch_id + 1)

            # 5. Add to summary writer tensorboard
            if iteration % REPORT_EVERY_ITER == 0:

                # save only if necessary (generator trained well)
                
                mean_autoencoder_ref_loss = np.mean(autoencoder_ref_losses)
                mean_autoencoder_mask_loss = np.mean(autoencoder_mask_losses)
                mean_discriminator_loss = np.mean(discriminator_losses)

                log.info("Iteration %d: autoencoder_ref_loss=%.3e, autoencoder_mask_loss=%.3e, discriminator_loss=%.3e, discriminator_accuracy=%.3f, discriminator_auc_roc=%.3f", iteration, mean_autoencoder_ref_loss, mean_autoencoder_mask_loss, mean_discriminator_loss, discriminator_accuracy, discriminator_auc_roc)
                # log.info("Iteration %d: autoencoder_ref_loss=%.3e, autoencoder_mask_loss=%.3e, discriminator_loss=%.3e, discriminator_accuracy=%.3f", iteration, mean_autoencoder_ref_loss, mean_autoencoder_mask_loss, mean_discriminator_loss, discriminator_accuracy)
                
                writer.add_scalar("autoencoder_ref_loss", mean_autoencoder_ref_loss, iteration)
                writer.add_scalar("autoencoder_mask_loss", mean_autoencoder_mask_loss, iteration)
                writer.add_scalar("epoch", epoch, iteration)

                # save only if necessary (generator trained well)
                writer.add_scalar("discriminator_loss", mean_discriminator_loss, iteration)
                writer.add_scalar("discriminator_acc", discriminator_accuracy, iteration)
                writer.add_scalar("discriminator_auc_roc", discriminator_auc_roc, iteration)

                autoencoder_ref_losses = []
                autoencoder_mask_losses = []
                discriminator_losses = []
                
            if iteration % SAVE_IMAGE_EVERY_ITER == 0:

                #writer.add_image("noisy", vutils.make_grid(batch_inputs[:IMAGE_SIZE], normalize=True), iteration)
                
                # TODO: improve this part
                echannels = 3 # expected images chanels

                n_input_channels = list(batch_input.size())[1]

                writer.add_image(f"Most noisy", vutils.make_grid(batch_input[:NB_IMAGES, 0:echannels], normalize=False), iteration)
                writer.add_image(f"Last denoised", vutils.make_grid(torch_reference_sequence.data[:NB_IMAGES, n_input_channels-echannels:n_input_channels], normalize=False), iteration)
                writer.add_image(f"Real", vutils.make_grid(batch_ref_data.data[:NB_IMAGES], normalize=False), iteration)

                writer.add_image(f"Ref mask", vutils.make_grid(ref_mask.data[:NB_IMAGES], normalize=False), iteration)
                writer.add_image(f"Noisy mask", vutils.make_grid(noisy_mask.data[:NB_IMAGES], normalize=False), iteration)

                # cumulative_channel += c

            # 6. Backup models information
            if iteration % BACKUP_EVERY_ITER == 0:
                
                epoch_str = str(epoch)

                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str

                save_models_folder_path = os.path.join(BACKUP_FOLDER, p_save, epoch_str)

                save_global_model_path = os.path.join(save_models_folder_path, BACKUP_MODEL_NAME.format('global'))
                save_discriminator_model_path = os.path.join(save_models_folder_path, BACKUP_MODEL_NAME.format('discriminator'))
                save_autoencoder_ref_model_path = os.path.join(save_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder_ref'))
                save_autoencoder_mask_model_path = os.path.join(save_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder_mask'))
                
                if not os.path.exists(save_models_folder_path):
                    os.makedirs(save_models_folder_path)

                torch.save({
                    'iteration': iteration,
                    'autoencoder_ref_state_dict': autoencoder_ref.state_dict(),
                    'optimizer_ref_state_dict': autoencoder_ref_optimizer.state_dict(),
                    'autoencoder_ref_losses': autoencoder_ref_losses
                }, save_autoencoder_ref_model_path)

                torch.save({
                    'iteration': iteration,
                    'autoencoder_mask_state_dict': autoencoder_mask.state_dict(),
                    'optimizer_mask_state_dict': autoencoder_mask_optimizer.state_dict(),
                    'autoencoder_mask_losses': autoencoder_mask_losses
                }, save_autoencoder_mask_model_path)

                # save only if necessary (generator trained well)
                
                torch.save({
                            'model_state_dict': discriminator.state_dict(),
                            'optimizer_state_dict': discriminator_optimizer.state_dict(),
                            'discriminator_losses': discriminator_losses
                    }, save_discriminator_model_path)

                torch.save({
                            'iteration': iteration,
                            'epochs': epoch
                        }, save_global_model_path)

            # 7. increment number of iteration
            iteration += 1
        
        if epoch >= start_epoch:
            writer.add_scalar("epoch", epoch + 1, iteration)

if __name__ == "__main__":
    main()