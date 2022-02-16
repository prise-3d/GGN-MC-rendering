# Guided-Generative Network for Monte-Carlo rendering

## Description

The proposed Guided-Generative Network (GGN) is applied on photorealistic images which are rendered by Monte-Carlo methods by evaluating a large number of samples per pixel. An insufficient number of samples per pixel tends to result in residual noise which is very noticeable to humans. This noise can be reduced by increasing the number of samples, as proven by Monte-Carlo theory, but this involves considerable computational time. Finding the right number of samples needed for human observers to perceive no noise is still an open problem.

GGN architecture whose purpose is to obtain automatic noise-related features is composed of $3$ neural network models:
- **Denoiser:** U-Net based model which denoises each image of different level of noise from a sliding-window;
- **Feature Map Generator**: an autoencoder model which takes as input an image sliding window and produces a noise feature map (NFM);
- **Discriminator**: classical CNN model for binary classification task which takes as input two NFM, one from the input sliding window and one from the denoised sliding window.

<img src="resources/images/total_gan_scheme.svg">

## Installation

```bash
git clone https://github.com/prise-3d/Guided-Generative-Newtork-MC-rendering.git
```

```bash
pip install -r requirements.txt
```

## Data preparation 

You can download the available dataset following this [link](https://prise3d.univ-littoral.fr/resources/sin3d/).

**Description of the dataset:**

This image base proposes several points of view where for each, images of different sample levels have been saved. Human thresholds have been collected to specify at what level of noise the image appears to be of good quality. These thresholds were collected for non-overlapped blocks of size `200 x 200` from the `800 x 800` pixel image.


Prepare the input data for the model:
```bash
python processing/generate_dataset_ggn.py --dataset /path/to/SIN3D-dataset --thresholds resources/human-thresholds.csv --scenes ~/data/nd-selected-scened.csv --sequence 6 --nb 500 --tile_size "200,200" --output /path/to/output
```

**Parameters indications:**
- `sequence`: specify the sliding window size;
- `tile_size`: specify the expected blocks size inside the image of `800 x 800` pixels;
- `nb`: the number of patch (noise level) extracted from the current block.

## Training the model

## Paper and citation

- Paper : [download](https://hal.archives-ouvertes.fr/hal-03374214v1)

Cite this paper:
```
@inproceedings{9680095,
  author={Buisine, Jérôme and Teytaud, Fabien and Delepoulle, Samuel and Renaud, Christophe},
  booktitle={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Guided-Generative Network for noise detection in Monte-Carlo rendering}, 
  year={2021},
  volume={},
  number={},
  pages={61-66},
  doi={10.1109/ICMLA52953.2021.00018}}
```

## License

[MIT](LICENSE)