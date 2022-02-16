# main imports
import argparse
import numpy as np
import os, sys

# image imports
from PIL import Image
from ipfml.processing.segmentation import divide_in_blocks

# others imports
import math
import random

# important variables
data_train_folder  = 'train'
data_test_folder   = 'test'

data_ref_folder    = 'references'

number_of_images   = 0 # used for writing extraction progress 
images_counter     = 0 # counter used for extraction progress

zones = np.arange(16)
h_zone, w_zone = (200, 200)
samples_step = 20
rotation_choices = np.arange(1, 4) # number of times to rotate

'''
Display progress information as progress bar
'''
def write_progress(progress):
    barWidth = 150

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

def min_max_norm(tile, image):
    return (tile - image.min()) / (image.max() - image.min())
    
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

# TODO: add preprocessing here
def preprocess_block(block):

    return block

'''
Constuct all dataset with specific tile size
'''
def construct_tiles(scene, dataset_path, thresholds, output_path, nb, tile_size, sequence, set_path):

    images_counter = 0

    h_tile, w_tile = tile_size

    # compute output reference folder
    output_folder = os.path.join(output_path, set_path)

    # map will store references combined data 
    # map will store references combined data 
    ref_output_path = os.path.join(output_folder, data_ref_folder)
    input_output_path = os.path.join(output_folder, 'input')
    weights_output_path = os.path.join(output_folder, 'weights')
    label_output_path = os.path.join(output_folder, 'labels')

    # concat all outputs folder
    output_folders = [ref_output_path, input_output_path, weights_output_path, label_output_path]

    output_img_index = 0

    print(f'Load of "{set_path}" scene: {scene} images')
    # build path if necessary
    for output_folder in output_folders:
        scene_output_folder = os.path.join(output_folder, scene)
        if not os.path.exists(scene_output_folder):
            os.makedirs(scene_output_folder)

    # store scene zones images
    scenes_images = {}

    for z in zones:
        scenes_images[z] = []

    scene_path = os.path.join(dataset_path, scene)

    images_names = sorted(os.listdir(scene_path))

    # read all images of scene
    for img in images_names:
        img_path = os.path.join(scene_path, img)
        img_data = np.array(Image.open(img_path))
        img_data = np.array(img_data, 'float16') / 255.

        blocks = divide_in_blocks(img_data, (h_zone, w_zone), pil=False)

        # store each image blocks for scene
        for z in zones:
            block = preprocess_block(blocks[z])
            scenes_images[z].append(block)

    for z in zones:
        print(f' -- zone {z}: {len(scenes_images[z])}')

    # check already generated data
    max_expected = len(zones) * nb
    scene_path = os.path.join(ref_output_path, scene)
    n_generated = len(os.listdir(scene_path))
    print('-----------------------')
    print(f' -- Already generated for {scene}: {n_generated}')

    if n_generated >= max_expected:
        print(f' ==> all data already generated for {scene}')
        return 

    for z in zones:
        
        zone_human_threshold = thresholds[scene][z]

        # get here number of images under threshold and over threshold
        thresholds_index = int(zone_human_threshold / samples_step) - 1

        # number of non noisy images (over thresholds)
        n_zones_images = len(scenes_images[z])


        current_blocks = scenes_images[z]
        
        reference_block = scenes_images[z][-1]

        # generate input model data:
        # Random noise block level under thresholds (noisy patch)
        # Random noise block level over human threshold (not noisy patch)
        # Random patch position over `zone` size
        if n_generated >= (z + 1) * nb:
            output_img_index += nb
            continue
        elif n_generated >= z * nb and n_generated < (z + 1) * nb:
            output_img_index += n_generated - z * nb
            counter_level = n_generated - z * nb
        else:
            counter_level = 0

        print(f' --- Start from {counter_level} for zone {z}')

        for _ in range(nb - counter_level):
                
            # ensure at least once each level of noise
            if counter_level < n_zones_images:
                random_input_index = counter_level
                counter_level += 1
            else:
                counter_level = 0 # restart

            # random_noisy_index = random.randint(0, thresholds_index)

            # specific random rotation
            # rotation = np.random.choice(rotation_choices)
            rotation = 0
            
            # compute output image name (patch image name)
            output_index_str = str(output_img_index)

            while len(output_index_str) < 11:
                output_index_str = '0' + output_index_str

            output_image_name = scene + '_' + output_index_str + '.npy'

            # by default
            if h_tile >= h_zone:
                h_random = 0
                w_random = 0
            else:
                h_random = random.randint(0, h_zone - h_tile - 1)
                w_random = random.randint(0, w_zone - w_tile - 1)

            h_end = h_random+h_tile
            w_end = w_random+w_tile

            # check if this image will be saved in to test or training dataset
            # p = random.random()

            # Add here sequence data
            tile_extract_input = None

            # check out of label
            if random_input_index + sequence > n_zones_images:
                random_input_index = n_zones_images - sequence
                counter_level = 0
        
            # create sequence of data
            for i in range(sequence):
                
                # go to next index and find tile size
                input_block = current_blocks[random_input_index + i][h_random:h_end, w_random:w_end]

                # add block rotation
                input_block = np.rot90(np.array(input_block, 'float16'), k=rotation)

                # create noisy block sequence
                if tile_extract_input is None:

                    if input_block.ndim < 3:
                        tile_extract_input = input_block[:, :, np.newaxis]
                    else:
                        tile_extract_input = input_block
                else:
                    tile_extract_input = np.concatenate((tile_extract_input, input_block), axis=2)


            current_input_nsamples = (random_input_index + sequence) * samples_step

            current_weight = 1. / math.log(current_input_nsamples)
            current_label = int(random_input_index < thresholds_index)

            # patch for input block
            input_path = os.path.join(input_output_path, scene, output_image_name)
            np.save(input_path, tile_extract_input)

            # patch for reference
            tile_extract_ref = reference_block[h_random:h_end, w_random:w_end]
            
            output_reference_path = os.path.join(ref_output_path, scene, output_image_name)
            np.save(output_reference_path, np.rot90(np.array(tile_extract_ref, 'float16'), k=rotation))

            output_weight_path = os.path.join(weights_output_path, scene, output_image_name)
            np.save(output_weight_path, np.array(current_weight, 'float16'))

            output_label_path = os.path.join(label_output_path, scene, output_image_name)
            np.save(output_label_path, np.array(current_label, 'uint8'))

            output_img_index = output_img_index + 1
            # images_counter = images_counter + 1

    # write progress using global variable
    # write_progress((images_counter + 1) / number_of_images)
        
    # del scenes_images

def main():

    global number_of_images, images_counter

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--dataset', type=str, help="dataset will all scenes images", required=True)
    parser.add_argument('--thresholds', type=str, help='thresholds data information', required=True)
    parser.add_argument('--scenes', type=str, help='expected selected scenes', required=True)
    parser.add_argument('--sequence', type=int, help='expected sequence size', required=True)
    parser.add_argument('--nb', type=int, help='number of tile extracted from each images and each zone', required=True)
    parser.add_argument('--tile_size', type=str, help='specify size of the tile used', default='32,32')
    parser.add_argument('--output', type=str, help='output folder of whole data `test` and `train` folder', required=True)

    args = parser.parse_args()

    p_dataset       = args.dataset
    p_thresholds    = args.thresholds
    p_scenes        = args.scenes
    p_sequence      = args.sequence
    p_nb            = args.nb
    p_tile          = args.tile_size.split(',')
    p_output        = args.output

    scenes_list = []

    with open(p_scenes) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            scene = data[0]

            scenes_list.append(scene)

    tile_size = int(p_tile[0]), int(p_tile[1])

    thresholds = extract_thresholds_from_file(p_thresholds)

    # get list scenes folders and shuffle it
    random.shuffle(scenes_list)

    number_of_images = sum([ len(zones) * p_nb for scene in scenes_list ]) # get total number of images from first feature path
    # print(thresholds)
    print('------------------------------------------------------------------------------------------------------')
    print('-- Start generating data')
    print('------------------------------------------------------------------------------------------------------')

    # print(number_of_images)
    # contruct tiles
    for scene in thresholds.keys():

        if scene in scenes_list:
            construct_tiles(scene, p_dataset, thresholds, p_output, p_nb, tile_size, p_sequence, 'train')
        else:
            construct_tiles(scene, p_dataset, thresholds, p_output, p_nb, tile_size, p_sequence, 'test')

    print()
    

if __name__ == "__main__":
    main()