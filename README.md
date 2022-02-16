# Guided-Generative Network for stopping criterion in Monte-Carlo rendering


## Description

## Installation

```bash
git clone https://github.com/prise-3d/Guided-Generative-Newtork-MC-rendering.git
```

```bash
pip install -r requirements.txt
```

## Data preparation 

You can download the available dataset following this [link](https://prise3d.univ-littoral.fr/resources/sin3d/).


Then, you can prepare the generated data by adding different levels of noise from specific scenes:
- 20 samples
- 40 samples
- 60 samples
- 
```bash
mkdir data && mkdir data/generated
python processing/extract_specific_png.py --scenes resources/selected_scenes.csv --folder path/to/SIN3D_dataset --index 20 --output data/generated/SIN3D_inputs
python processing/extract_specific_png.py --scenes resources/selected_scenes.csv --folder path/to/SIN3D_dataset --index 40 --output data/generated/SIN3D_inputs
python processing/extract_specific_png.py --scenes resources/selected_scenes.csv --folder path/to/SIN3D_dataset --index 60 --output data/generated/SIN3D_inputs
```

And get expected references:
```bash
python processing/extract_specific_png.py --folder path/to/SIN3D_dataset --index 10000 --output data/generated/SIN3D_references
```

if references images are human thresholds reconstructed images:

```bash
python processing/reconstruct_images_human_thresholds.py --folder path/to/SIN3D_dataset --thresholds resources/thresholds_SVD-Entropy_v2.csv --output data/human_references
```

## Start training

## Citation

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