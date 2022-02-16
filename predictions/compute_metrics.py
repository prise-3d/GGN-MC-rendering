# main imports
import numpy as np
import argparse
import os
import math
import sys

# image processing
from PIL import Image
from ipfml.processing.segmentation import divide_in_blocks

# deep learning imports
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

# other parameters
BACKUP_MODEL_NAME = "{}_model.pt"
BACKUP_FOLDER = "saved_models"

zones = np.arange(16)
zone_size = (200, 200)
samples_step = 20

# models imports
sys.path.insert(0, '') # trick to enable import of main folder module
from models.seq_processing.model_choice import prepare_model, choices_input

'''
Display progress information as progress bar
'''
def write_progress(progress):
    barWidth = 120

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")

def extract_thresholds_from_file(filename):
    # extract thresholds
    thresholds_dict = {}
    with open(filename) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            thresholds_dict[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

    return thresholds_dict

def main():

    parser = argparse.ArgumentParser(description="Denoise folder of image using Autoencoder")

    parser.add_argument('--dataset', type=str, help="path of dataset to use", required=True)
    parser.add_argument('--thresholds', type=str, help="path of the human thresholds", required=True)
    parser.add_argument('--scenes', type=str, help='csv file with selected scenes', required=True)
    parser.add_argument('--tile_size', type=str, help='specify size of the tile used for model', default='50,50')
    parser.add_argument('--sequence', type=int, help='sequence size', default=1)
    parser.add_argument('--seqlayer', type=str, help='sequence layer input', default="1")
    parser.add_argument('--choice', type=str, help='input data format', default=choices_input[0], choices=choices_input)
    parser.add_argument('--load', type=str, help='folder backup model', required=True)
    parser.add_argument('--output', type=str, help='csv output filename', required=True)

    args = parser.parse_args()

    p_dataset      = args.dataset
    p_thresholds   = args.thresholds
    p_scenes       = args.scenes
    p_tile         = args.tile_size.split(',')
    p_sequence     = args.sequence
    p_seqlayer     = list(map(int, args.seqlayer.split(',')))
    p_choice       = args.choice
    p_load         = args.load
    p_output       = args.output

    # thresholds extraction
    thresholds = extract_thresholds_from_file(p_thresholds)

    tile_size = int(p_tile[0]), int(p_tile[1])
    n_sub_blocks = int(zone_size[0] / tile_size[0]) * int(zone_size[1] / tile_size[1])

    scenes_list = []

    with open(p_scenes, 'r') as f:
        for l in f.readlines():
            scenes_list.append(l.split(';')[0])

    print('[1] -- Load of Discriminator model...')

    # creating and loading model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # define models and loss functions
    # by default fixed autoencoder size
    autoencoder_ref, autoencoder_mask, discriminator = prepare_model(p_choice, p_seqlayer, tile_size[0], p_sequence)

    # prepare folder names to save models
    load_models_folder_path = os.path.join(BACKUP_FOLDER, p_load)

    load_discriminator_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('discriminator'))
    load_autoencoder_ref_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder_ref'))
    load_autoencoder_mask_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder_mask'))

    # load autoencoder and discriminator state
    discriminator_checkpoint = torch.load(load_discriminator_model_path, map_location=device)
    autoencoder_ref_checkpoint = torch.load(load_autoencoder_ref_model_path, map_location=device)
    autoencoder_mask_checkpoint = torch.load(load_autoencoder_mask_model_path, map_location=device)

    discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
    discriminator.eval()

    autoencoder_ref.load_state_dict(autoencoder_ref_checkpoint['autoencoder_ref_state_dict'])
    autoencoder_ref.eval()

    autoencoder_mask.load_state_dict(autoencoder_mask_checkpoint['autoencoder_mask_state_dict'])
    autoencoder_mask.eval()

    acc_scores = []
    auc_scores = []

    print(f'[2] -- Start prediction over {len(scenes_list)} scenes...')
    for s_i, scene in enumerate(sorted(scenes_list)):
        
        scenes_images = {}

        # store all expected prediction for all zones of the image
        expected_predictions = []
        model_predictions = []

        for z in zones:
            scenes_images[z] = []

        print('----')
        print(f'    -- Load data for scene {s_i + 1} of {len(scenes_list)}...')

        scene_path = os.path.join(p_dataset, scene)

        images_names = sorted(os.listdir(scene_path))

        # read all images of scene
        for img in images_names:
            img_path = os.path.join(scene_path, img)
            img_data = np.array(Image.open(img_path))

            blocks = divide_in_blocks(img_data, zone_size, pil=False)

            # store each image blocks for scene
            for z in zones:
                scenes_images[z].append(np.array(blocks[z] / 255., 'float32'))

        
        print(f'    -- Predict thresholds for scene {scene} [{s_i + 1} of {len(scenes_list)}]')
        thresholds_found = []
        all_predictions = []

        ncounter = 0
        max_counters = len(zones) * len(images_names)

        # for each zone load each data
        for z in zones:

            sub_thresholds = {}
            sub_thresholds_sequence = {}
            current_zone_predictions = []

            for i in range(n_sub_blocks):
                sub_thresholds[i] = None
                sub_thresholds_sequence[i] = []

            # loaded zone data to inspect using model
            for b_i, block in enumerate(scenes_images[z]):
                sub_blocks = divide_in_blocks(block, tile_size, pil=False)

                nsamples = (b_i + 1) * samples_step

                # for each sub blocks, call model and get probability
                for sb_i, sblock in enumerate(sub_blocks):

                    # remove first block if max sequence size is already reached
                    if len(sub_thresholds_sequence[sb_i]) >= p_sequence:
                        sub_thresholds_sequence[sb_i].pop(0)

                    # add new block
                    # if len(sub_thresholds_sequence[sb_i]) < p_sequence:
                    sub_thresholds_sequence[sb_i].append(sblock)

                    if len(sub_thresholds_sequence[sb_i]) >= p_sequence:
                        
                        # prepare data ref and input for mask autoencoder   
                        prepared_input_sequence = []
                        prepared_ref_sequence = []

                        for cblock in sub_thresholds_sequence[sb_i]:
                            
                            moved_input = np.moveaxis(cblock, -1, 0)

                            prepared_input = np.expand_dims(moved_input, axis=0)
                            torch_input = torch.from_numpy(prepared_input).float()

                            prepared_input_sequence.append(torch_input)

                            predicted_ref = autoencoder_ref(torch.from_numpy(prepared_input).float())
                            prepared_ref_sequence.append(predicted_ref)

                        prepared_input_sequence = torch.cat(prepared_input_sequence, dim=1)
                        predicted_ref_sequence = torch.cat(prepared_ref_sequence, dim=1)

                        # Get data mask from sequence
                        ref_mask = autoencoder_mask(predicted_ref_sequence)
                        noisy_mask = autoencoder_mask(prepared_input_sequence)

                        inputs_mask = torch.cat((ref_mask, noisy_mask), dim=1)

                        # predict output prob
                        prob = discriminator(inputs_mask)
                        current_prob = prob.detach().numpy()[0]
                        # check if not noisy (check if fake)
                        # noisy => 1 (reference)
                        # not noisy => 0

                        if current_prob < 0.5: # TODO : check expected thresholds
                            
                            if sub_thresholds[sb_i] is None:
                                sub_thresholds[sb_i] = nsamples

                        model_predictions.append(current_prob)
                        current_zone_predictions.append(current_prob)

                        if nsamples > thresholds[scene][z]:
                            expected_predictions.append(0)
                        else:
                            expected_predictions.append(1)

                write_progress((ncounter + 1) / max_counters)
                ncounter += 1

            for i in range(n_sub_blocks):
                if sub_thresholds[i] is None:
                    sub_thresholds[i] = samples_step * len(scenes_images[z])

            # save current zone predictions and label
            thresholds_found.append(np.max([ sub_thresholds[i] for i in range(n_sub_blocks)]))
            all_predictions.append(current_zone_predictions)
        
        print(f'Thresholds found: {thresholds_found}')

        # TODO: print and save metrics for current scene
        binary_predictions = list(map(lambda x: 1 if x > 0.5 else 0, model_predictions))
        accurary_model_score = accuracy_score(expected_predictions, binary_predictions)
        roc_auc_model_score = roc_auc_score(expected_predictions, model_predictions)

        acc_scores.append(accurary_model_score)
        auc_scores.append(roc_auc_model_score)

        print(f'\n ---- Model performs on {scene}: {{acc:{accurary_model_score}, auc:{roc_auc_model_score}}}')
        
        if not os.path.exists(p_output):
            os.makedirs(p_output)

        with open(os.path.join(p_output, 'metrics.csv'), 'a') as f:
            f.write(f'{scene};{accurary_model_score};{roc_auc_model_score}\n')

        with open(os.path.join(p_output, 'thresholds.csv'), 'a') as f:
            f.write(f'{scene}')

            for i in range(len(zones)):
                f.write(f';{thresholds_found[i]}')
            f.write('\n')

        with open(os.path.join(p_output, 'predictions.csv'), 'a') as f:

            for i in range(len(zones)):
                f.write(f'{scene};{i}')

                for v in all_predictions[i]:
                    f.write(f';{v}')
                        
                f.write('\n')

    print('\n -- Model mean performs: {{acc:{np.mean(acc_scores)}, auc:{np.mean(auc_scores)}}}')

if __name__ == "__main__":
    main()