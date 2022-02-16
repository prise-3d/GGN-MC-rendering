from .prepare_data import compute_svd_reconstruction, compute_noise_mask
from ipfml.processing.transform import get_LAB_L_SVD_s

import numpy as np
import torch

def compute_input_data(choice, predicted, sequence_data, device):

    if choice == 'gaussian_mask_var':
        
        predicted_mask = compute_noise_mask(predicted)

        seq_data = []

        for w in range(len(sequence_data)):

            current_image = sequence_data[w]

            current_image_mask = compute_noise_mask(current_image)

            # get current absolute difference
            seq_data.append(np.abs(predicted_mask - current_image_mask))
        
        seq_data = np.var(np.array(seq_data), axis=0)
        seq_data = seq_data / (np.max(seq_data, axis=0) + 0.0000001)

        return seq_data[np.newaxis, :, :]

    if choice == 'gaussian_mask':
        
        predicted_mask = compute_noise_mask(predicted)

        seq_data = []

        for w in range(len(sequence_data)):

            current_image = sequence_data[w]

            current_image_mask = compute_noise_mask(current_image)

            # get current absolute difference
            seq_data.append(np.abs(predicted_mask - current_image_mask))
        
        seq_data = np.array(seq_data)
        # seq_data = seq_data / (np.max(seq_data, axis=0) + 0.0000001)

        return seq_data

    if choice == 'svd_reconstruction_var':
        
        predicted_svd = compute_svd_reconstruction(predicted)

        seq_data = []

        for w in range(len(sequence_data)):

            current_image = sequence_data[w]

            current_image_svd = compute_svd_reconstruction(current_image)

            # get current absolute difference
            seq_data.append(np.abs(predicted_svd - current_image_svd))
        
        seq_data = np.var(np.array(seq_data), axis=0)
        seq_data = seq_data / (np.max(seq_data, axis=0) + 0.0000001)

        return seq_data[np.newaxis, :, :]

    if choice == 'svd_reconstruction':
        
        predicted_svd = compute_svd_reconstruction(predicted)

        seq_data = []

        for w in range(len(sequence_data)):

            current_image = sequence_data[w]

            current_image_svd = compute_svd_reconstruction(current_image)

            # get current absolute difference
            seq_data.append(np.abs(predicted_svd - current_image_svd))
        
        seq_data = np.concatenate(seq_data, axis=0)
        # seq_data = seq_data / (np.max(seq_data, axis=0) + 0.0000001)

        return seq_data

    if choice == 'svd_diff_rnn':
        
        predicted_svd = get_LAB_L_SVD_s(predicted)

        seq_data = []

        for w in range(len(sequence_data)):

            current_image = sequence_data[w]

            current_image_svd = get_LAB_L_SVD_s(current_image)

            # get current absolute difference
            seq_data.append(np.abs(predicted_svd - current_image_svd))

        return np.array(seq_data / np.max(seq_data, axis=0))

    if choice == 'sv_var':
        
        predicted_svd = get_LAB_L_SVD_s(predicted)

        seq_data = []

        for w in range(len(sequence_data)):

            current_image = sequence_data[w]

            current_image_svd = get_LAB_L_SVD_s(current_image)

            # get current absolute difference
            seq_data.append(np.abs(predicted_svd - current_image_svd))

        return np.var(np.array(seq_data), axis=0)

    if choice == 'sv_seq':
        
        predicted_svd = get_LAB_L_SVD_s(predicted)

        seq_data = []

        for w in range(len(sequence_data)):

            current_image = sequence_data[w]

            current_image_svd = get_LAB_L_SVD_s(current_image)

            # get current absolute difference
            seq_data.append(np.abs(predicted_svd - current_image_svd))

        return np.array(seq_data / np.max(seq_data, axis=0))

    # default return...
    return None


def compute_input_train_data(choice, predicted_batch, noisy_sequence_batch, not_noisy_sequence_batch, p_sequence, device, channels=3):

    if torch.cuda.is_available():
        noisy_sequence_batch = noisy_sequence_batch.cpu().numpy()
        not_noisy_sequence_batch = not_noisy_sequence_batch.cpu().numpy()
        predicted_batch = predicted_batch.cpu().numpy()
    else:
        noisy_sequence_batch = noisy_sequence_batch.numpy()
        not_noisy_sequence_batch = not_noisy_sequence_batch.cpu().numpy()
        predicted_batch = predicted_batch.numpy()

    b_size, _, _, _ = noisy_sequence_batch.shape

    noisy_batch_data = []
    not_noisy_batch_data = []

    for i in range(b_size):
        
        # compute for different data choices
        if choice == 'gaussian_mask_var':
            
            predicted_mask = compute_noise_mask(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            noisy_seq_data = []

            for w in range(p_sequence):

                current_image = noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_mask = compute_noise_mask(current_image.transpose(1, 2, 0))

                # get current absolute difference
                noisy_seq_data.append(np.abs(predicted_mask - current_image_mask))
            
            noisy_seq_data = np.var(np.array(noisy_seq_data), axis=0)
            noisy_seq_data = noisy_seq_data / (np.max(noisy_seq_data, axis=0) + 0.0000001)

            noisy_batch_data.append(noisy_seq_data[np.newaxis, :, :])

            # compute for not noisy information
            not_noisy_seq_data = []

            for w in range(p_sequence):

                current_image = not_noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_mask = compute_noise_mask(current_image.transpose(1, 2, 0))

                # get current absolute difference
                not_noisy_seq_data.append(np.abs(predicted_mask - current_image_mask))
            
            not_noisy_seq_data = np.var(np.array(not_noisy_seq_data), axis=0)
            not_noisy_seq_data = not_noisy_seq_data / (np.max(not_noisy_seq_data, axis=0) + 0.0000001)

            not_noisy_batch_data.append(not_noisy_seq_data[np.newaxis, :, :])

        if choice == 'gaussian_mask':
            
            predicted_mask = compute_noise_mask(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            noisy_seq_data = []

            for w in range(p_sequence):

                current_image = noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_mask = compute_noise_mask(current_image.transpose(1, 2, 0))

                # get current absolute difference
                noisy_seq_data.append(np.abs(predicted_mask - current_image_mask))
            
            noisy_seq_data = np.array(noisy_seq_data)
            # noisy_seq_data = noisy_seq_data / (np.max(noisy_seq_data, axis=0) + 0.0000001)

            noisy_batch_data.append(noisy_seq_data)

            # compute for not noisy information
            not_noisy_seq_data = []

            for w in range(p_sequence):

                current_image = not_noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_mask = compute_noise_mask(current_image.transpose(1, 2, 0))

                # get current absolute difference
                not_noisy_seq_data.append(np.abs(predicted_mask - current_image_mask))
            
            not_noisy_seq_data = np.array(not_noisy_seq_data)
            # not_noisy_seq_data = not_noisy_seq_data / (np.max(not_noisy_seq_data, axis=0) + 0.0000001)

            not_noisy_batch_data.append(not_noisy_seq_data)


        if choice == 'svd_reconstruction_var':
            
            predicted_svd = compute_svd_reconstruction(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            noisy_seq_data = []

            for w in range(p_sequence):

                current_image = noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = compute_svd_reconstruction(current_image.transpose(1, 2, 0))

                # get current absolute difference
                noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))
            
            noisy_seq_data = np.var(np.array(noisy_seq_data), axis=0)
            noisy_seq_data = noisy_seq_data / (np.max(noisy_seq_data, axis=0) + 0.0000001)

            noisy_batch_data.append(noisy_seq_data)

            # compute for not noisy information
            not_noisy_seq_data = []

            for w in range(p_sequence):

                current_image = not_noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = compute_svd_reconstruction(current_image.transpose(1, 2, 0))

                # get current absolute difference
                not_noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))
            
            not_noisy_seq_data = np.var(np.array(not_noisy_seq_data), axis=0)
            not_noisy_seq_data = not_noisy_seq_data / (np.max(not_noisy_seq_data, axis=0) + 0.0000001)

            not_noisy_batch_data.append(not_noisy_seq_data)

        if choice == 'svd_reconstruction':
            
            predicted_svd = compute_svd_reconstruction(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            noisy_seq_data = []

            for w in range(p_sequence):

                current_image = noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = compute_svd_reconstruction(current_image.transpose(1, 2, 0))

                # get current absolute difference
                noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))
            
            noisy_seq_data = np.concatenate(noisy_seq_data, axis=0)
            # noisy_seq_data = noisy_seq_data / (np.max(noisy_seq_data, axis=0) + 0.0000001)

            noisy_batch_data.append(noisy_seq_data)

            # compute for not noisy information
            not_noisy_seq_data = []

            for w in range(p_sequence):

                current_image = not_noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = compute_svd_reconstruction(current_image.transpose(1, 2, 0))

                # get current absolute difference
                not_noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))
            
            not_noisy_seq_data = np.concatenate(not_noisy_seq_data, axis=0)
            # not_noisy_seq_data = not_noisy_seq_data / (np.max(not_noisy_seq_data, axis=0) + 0.0000001)

            not_noisy_batch_data.append(not_noisy_seq_data)

        if choice == 'svd_diff_rnn':
            
            predicted_svd = get_LAB_L_SVD_s(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            noisy_seq_data = []

            for w in range(p_sequence):

                current_image = noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))
            

            noisy_batch_data.append(np.array(noisy_seq_data) / np.max(noisy_seq_data, axis=0))

            # compute for not noisy information
            not_noisy_seq_data = []

            for w in range(p_sequence):

                current_image = not_noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                not_noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))

            not_noisy_batch_data.append(np.array(not_noisy_seq_data) / np.max(not_noisy_seq_data, axis=0))

        if choice == 'sv_var':
            
            predicted_svd = get_LAB_L_SVD_s(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            noisy_seq_data = []

            for w in range(p_sequence):

                current_image = noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))
            

            noisy_batch_data.append(np.var(np.array(noisy_seq_data), axis=0))

            # compute for not noisy information
            not_noisy_seq_data = []

            for w in range(p_sequence):

                current_image = not_noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                not_noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))

            not_noisy_batch_data.append(np.var(np.array(not_noisy_seq_data), axis=0))

        if choice == 'sv_seq':
            
            predicted_svd = get_LAB_L_SVD_s(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            noisy_seq_data = []

            for w in range(p_sequence):

                current_image = noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))
            

            noisy_batch_data.append(np.array(noisy_seq_data) / np.max(noisy_seq_data, axis=0))

            # compute for not noisy information
            not_noisy_seq_data = []

            for w in range(p_sequence):

                current_image = not_noisy_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                not_noisy_seq_data.append(np.abs(predicted_svd - current_image_svd))

            not_noisy_batch_data.append(np.array(not_noisy_seq_data) / np.max(not_noisy_seq_data, axis=0))


    noisy_batch_data = np.array(noisy_batch_data)
    t_noisy_batch_data = torch.tensor(noisy_batch_data, dtype=torch.float32, device=device)

    not_noisy_batch_data = np.array(not_noisy_batch_data)
    t_not_noisy_batch_data = torch.tensor(not_noisy_batch_data, dtype=torch.float32, device=device)
    
    return t_noisy_batch_data, t_not_noisy_batch_data

# redondant method...
def compute_input_sequence_data(choice, predicted_batch, input_sequence_batch, p_sequence, device, channels=3):

    if torch.cuda.is_available():
        input_sequence_batch = input_sequence_batch.cpu().numpy()
        predicted_batch = predicted_batch.cpu().numpy()
    else:
        input_sequence_batch = input_sequence_batch.numpy()
        predicted_batch = predicted_batch.numpy()

    b_size, _, _, _ = input_sequence_batch.shape

    input_batch_data = []

    for i in range(b_size):
        
        # compute for different data choices
        if choice == 'gaussian_mask_var':
            
            predicted_mask = compute_noise_mask(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            input_seq_data = []

            for w in range(p_sequence):

                current_image = input_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_mask = compute_noise_mask(current_image.transpose(1, 2, 0))

                # get current absolute difference
                input_seq_data.append(np.abs(predicted_mask - current_image_mask))
            
            input_seq_data = np.var(np.array(input_seq_data), axis=0)
            input_seq_data = input_seq_data / (np.max(input_seq_data, axis=0) + 0.0000001)

            input_batch_data.append(input_seq_data[np.newaxis, :, :])

        if choice == 'gaussian_mask':
            
            predicted_mask = compute_noise_mask(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            input_seq_data = []

            for w in range(p_sequence):

                current_image = input_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_mask = compute_noise_mask(current_image.transpose(1, 2, 0))

                # get current absolute difference
                input_seq_data.append(np.abs(predicted_mask - current_image_mask))
            
            input_seq_data = np.array(input_seq_data)

            input_batch_data.append(input_seq_data)


        if choice == 'svd_reconstruction_var':
            
            predicted_svd = compute_svd_reconstruction(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            input_seq_data = []

            for w in range(p_sequence):

                current_image = input_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = compute_svd_reconstruction(current_image.transpose(1, 2, 0))

                # get current absolute difference
                input_seq_data.append(np.abs(predicted_svd - current_image_svd))
            
            input_seq_data = np.var(np.array(input_seq_data), axis=0)
            input_seq_data = input_seq_data / (np.max(input_seq_data, axis=0) + 0.0000001)

            input_batch_data.append(input_seq_data)

        if choice == 'svd_reconstruction':
            
            predicted_svd = compute_svd_reconstruction(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            input_seq_data = []

            for w in range(p_sequence):

                current_image = input_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = compute_svd_reconstruction(current_image.transpose(1, 2, 0))

                # get current absolute difference
                input_seq_data.append(np.abs(predicted_svd - current_image_svd))
            
            input_seq_data = np.concatenate(input_seq_data, axis=0)
            
            input_batch_data.append(input_seq_data)

        if choice == 'svd_diff_rnn':
            
            predicted_svd = get_LAB_L_SVD_s(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            input_seq_data = []

            for w in range(p_sequence):

                current_image = input_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                input_seq_data.append(np.abs(predicted_svd - current_image_svd))
            

            input_batch_data.append(np.array(input_seq_data) / np.max(input_seq_data, axis=0))


        if choice == 'sv_var':
            
            predicted_svd = get_LAB_L_SVD_s(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            input_seq_data = []

            for w in range(p_sequence):

                current_image = input_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                input_seq_data.append(np.abs(predicted_svd - current_image_svd))
            

            input_batch_data.append(np.var(np.array(input_seq_data), axis=0))

        if choice == 'sv_seq':
            
            predicted_svd = get_LAB_L_SVD_s(predicted_batch[i].transpose(1, 2, 0))

            # compute for noisy information
            input_seq_data = []

            for w in range(p_sequence):

                current_image = input_sequence_batch[i, w*channels:w*channels+channels, :, :]

                current_image_svd = get_LAB_L_SVD_s(current_image.transpose(1, 2, 0))

                # get current absolute difference
                input_seq_data.append(np.abs(predicted_svd - current_image_svd))
            

            input_batch_data.append(np.array(input_seq_data) / np.max(input_seq_data, axis=0))

            
    input_batch_data = np.array(input_batch_data)
    t_input_batch_data = torch.tensor(input_batch_data, dtype=torch.float32, device=device)

    return t_input_batch_data