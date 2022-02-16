from models.autoencoders.ushaped import UShapedAutoencoder
from models.autoencoders.vae_conv_reduced import VAE
from models.autoencoders.ushaped_200 import UShapedAutoencoder as UShapedAutoencoder200
from models.autoencoders.autoencoder_200 import Autoencoder as Autoencoder200
from models.autoencoders.ushaped_200_reduced import UShapedAutoencoder as UShapedAutoencoder200_Reduced
from models.discriminators.discriminator_v1 import Discriminator as DiscriminatorV1
from models.discriminators.discriminator_200 import Discriminator as Discriminator200
from models.discriminators.discriminator_200_reduced import Discriminator as Discriminator200_Reduced
from models.discriminators.discriminator_seq import Discriminator as DiscriminatorLSTM
from models.discriminators.discriminator_ff import Discriminator as DiscriminatorFF
from models.discriminators.discriminator_ffs import Discriminator as DiscriminatorFFS

choices_input = ['svd_reconstruction_var', 'svd_reconstruction', 'gaussian_mask', 'gaussian_mask_var', 'svd_diff_rnn', 'sv_var', 'sv_seq', 'gen_mask', 'gen_ref_and_mask', 'gen_ref_and_mask_vae']

def prepare_model(choice, seq_layer, tile_size, sequence):

    autoencoder = None
    discriminator = None

    if choice == 'gaussian_mask' or choice == 'gaussian_mask_var' or choice == 'svd_reconstruction' or choice == 'svd_reconstruction_var':

        if tile_size > 100:
            discriminator = Discriminator200(sum(seq_layer), tile_size)
        else:
            discriminator = DiscriminatorV1(sum(seq_layer), tile_size)

        # RGB input for RGB output
        if tile_size > 100:
            autoencoder = UShapedAutoencoder200(3, 3, tile_size)
        else:
            autoencoder = UShapedAutoencoder(3, 3, tile_size)

    if choice == 'svd_diff_rnn':
        
        discriminator = DiscriminatorLSTM(tile_size, tile_size, 5, sequence)

        # RGB input for RGB output
        if tile_size > 100:
            autoencoder = UShapedAutoencoder200(3, 3, tile_size)
        else:
            autoencoder = UShapedAutoencoder(3, 3, tile_size)


    if choice == 'sv_var':
        
        discriminator = DiscriminatorFF(tile_size)

        # RGB input for RGB output
        if tile_size > 100:
            autoencoder = UShapedAutoencoder200(3, 3, tile_size)
        else:
            autoencoder = UShapedAutoencoder(3, 3, tile_size)


    if choice == 'sv_seq':
        
        discriminator = DiscriminatorFFS(tile_size, sequence)

        # RGB input for RGB output
        if tile_size > 100:
            autoencoder = UShapedAutoencoder200(3, 3, tile_size)
        else:
            autoencoder = UShapedAutoencoder(3, 3, tile_size)

    if choice == 'gen_mask':
        """Generation of custom mask using GAN
        """
        
        if tile_size > 100:
            discriminator = Discriminator200(1, tile_size)
        else:
            discriminator = DiscriminatorV1(1, tile_size)

        # RGB input sequence for Greay level output
        if tile_size > 100:
            autoencoder = UShapedAutoencoder200(3 * sequence, 1, tile_size)
        else:
            autoencoder = UShapedAutoencoder(3 * sequence, 1, tile_size)


    if choice == 'gen_ref_and_mask':
        """Generation of custom mask using GAN
        """
        
        # Two gray levels as input
        if tile_size > 100:
            discriminator = Discriminator200(2, tile_size)
        else:
            discriminator = DiscriminatorV1(2, tile_size)

        # RGB input for RGB output
        if tile_size > 100:
            autoencoder_ref = UShapedAutoencoder200_Reduced(3, 3, tile_size)
        else:
            autoencoder_ref = UShapedAutoencoder(3, 3, tile_size)

        # RGB input sequence for Greay level output
        if tile_size > 100:
            autoencoder_mask = Autoencoder200(3 * sequence, 1, tile_size)
        else:
            autoencoder_mask = Autoencoder200(3 * sequence, 1, tile_size)

        return autoencoder_ref, autoencoder_mask, discriminator

    if choice == 'gen_ref_and_mask_vae':
        """Generation of custom mask using GAN
        """
        
        # Two gray levels as input
        if tile_size > 100:
            discriminator = Discriminator200(2, tile_size)
        else:
            discriminator = DiscriminatorV1(2, tile_size)

        # RGB input for RGB output
        if tile_size > 100:
            autoencoder_ref = UShapedAutoencoder200_Reduced(3, 3, tile_size)
        else:
            autoencoder_ref = UShapedAutoencoder(3, 3, tile_size)

        # RGB input sequence for Greay level output
        if tile_size > 100:
            autoencoder_mask = VAE(3 * sequence, tile_size, 1, tile_size, 20)
        else:
            autoencoder_mask = UShapedAutoencoder(3 * sequence, 1, tile_size)

        return autoencoder_ref, autoencoder_mask, discriminator

    return autoencoder, discriminator