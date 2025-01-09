# 6DL-PoseGenerator



![Splash](./demo/demo.png)
## Overview
[![Published in SciTePress](https://img.shields.io/badge/SciTePress-Crane%20Spreader%20Pose%20Estimation%20from%20a%20Single%20View-green)](https://www.scitepress.org/PublishedPapers/2023/117888/117888.pdf)

The code presented on this repository was developed as an extension of the imlementation used in our paper ([Crane Spreader Pose Estimation from a Single View](https://www.scitepress.org/PublishedPapers/2023/117888/117888.pdf)). The rendering produced from this pipeline were used to train a 6D pose esimation DL algorithm with synthetic data. 

This repo utilizes OpenGL and custom shaders for rendering synthetic data of a given 3D Model (texture is optional) that can be used to train 6D pose estimation algorithms or even for 2D segmentation tasks. 

## Important features:
- [x] [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) compatible
- [x] Supports ply models with associated texture or RGB vertex colors (both optional)
- [x] Photorealistic rendering with lighting manipulation
- [x] Depth rendering with controlable depth scale


# Installation

```
git clone https://github.com/POSE-Lab/6DL-PoseGenerator
conda create -n dlpose python=3.10
conda activate dlpose
cd 6DL-PoseGenerator
pip install -r requirements.txt
```
# Usage
1. Modify the ```config.yaml``` according to your needs (See [Parameters](#parameters) section).
2. Run ```python render.py --io.config config.yaml``` .

You can find a full list of available parameters also from the CLI by typing ```python render.py --help```.

We also provide a script to visualize poses whether they are ground truth or estimated. The poses have to be in [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md). You can visualize poses by running:
```
python vis_poses.py \
--poses ./savePath/scene_gt.json \
--images ./savePath/texture/rgb \
--camera-params ./savePath/scene_camera.json \
--model-path ./demo/models/duck.ply \
--outPath ./vis
--opacity 0.7 # the opacity of the overlayed pose
```
You can view the availabe CLI arguments by running ```python vis_poses.py --help```.
# Parameters

Modify the ```config.yaml``` according to your needs, enter the model path and the outpath path. Some key parameters you can change include:

- General Rendering parameters:
    - FBO_WIDTH, FBO_HEIGHT: Resolution of rendered images.
    - phis, thetas, distances: Range of angles and distances to render the model.
    - background_color: The background color (no data value)
- Shader Associated parameters:
    - render_modes: The modes you want to render (```'triangles'``` = Rendering geometry without texture but with lighting, ```'texture'``` = Rendering of the textured model including ligting, ```'depth'```: Depth image renders).
    - texture_file: Path to texture file.

    **Note:** To use **RGB vertex colors** instead of texture for rendering set the ```texture_file = Null```. Otherwise, fill the path of the texture image.
- Lighting settings:
    - light_position: Position of the light. By default set to ```Null``` were the light coincides with the current camera position.
    - {triangles,texture}_{ambient,specular}_strength: Control Phong shading parameters.
    - triangles_object_color: Color of the mesh only when rendering in ```triangles``` mode.
- Pose associated parameters:
    - {rotation,translation}_perturbation: Introduce random rotation/translation to the poses for diverse poses distribution.
    - {rotation,translation}_{x,y,z}range: Magnititude of the random rotation/translations. Must me specified in the same unit as the object model.
    - depth_scale: Depth is saved as a 16-Bit Gray scale image. Thus the pixel value range from 0-65535. For depth_scale=1, the depth image is at 1-1 correspodence with the units of the model. E.g a pixel value of 100 means 100 units away from the camera. For small models its a good idea to increase the depth scale.
- Io associated parameters:
    - model_path: Path to the 3D model
    - object_id: Object id used only to write the ```scene_gt.json``` file. (See BOP format) 
    - savePath: Path to save renderings.


## Expected results

Your ```savePath``` path (set in ```config.yaml```) after rendering should have the following structure:

```
.
├── depth
├── geom
├── texture
├── scene_gt.json
└── scene_camera.json
```
- ```Depth``` folder containes the depth images.
- ```geom/rgb/``` contains the images rendered with ```triangles``` mode (See [Parameters](#parameters) section).
- ```texture/rgb/``` contains the images renderd with ```texture``` mode (See [Parameters](#parameters) section).
- ```scene_gt.json``` contains the poses in BOP format.
- ```scene_camera.json``` contains the camera intrisics in BOP format. 

## Demo example 1

For ```./demo/cube.ply``` the rendered results can be seen below.
Note that these images were produced with ```rotation_perturbation = False``` and ```rotation_perturbation = False```.
### Texture + Lighting
![Texture](./demo/montage_rgb.png)
### Geometry + Lighting
![Geom](./demo/montage_geom.png)
### Depth (Depth scale = 5000)
![Depth](./demo/montage_depth.png)


## Demo example 2

With ```rotation_perturbation = True``` and ```rotation_perturbation = True```.

![Texture_pert](./demo/montage_rgb_pert.png)

## Citation

If you find this code usefull please use the following BibTex to cite.

```bibtex
@article{pateraki_sapoutzoglou_lourakis_2023, 
    title={Crane Spreader Pose Estimation from a Single View}, 
    url={https://www.scitepress.org/PublishedPapers/2023/117888/117888.pdf}, 
    DOI={https://doi.org/10.5220/0011788800003417}, 
    journal={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications}, 
    publisher={SCITEPRESS - Science and Technology Publications}, 
    author={Pateraki, Maria and Sapoutzoglou, Panagiotis and Lourakis, Manolis}, 
    year={2023}, 
    pages={796–805}
}
```