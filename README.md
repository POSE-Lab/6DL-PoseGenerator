# 6DL-PoseGenerator

![Splash](./demo/demo.png)
## Overview

This repo utilizes OpenGL and custom shaders for rendering synthetic data of a given 3D Model (texture is optional) that can be used to train 6D pose estimation algorithms or even for 2D segmentation tasks. The structure of the code is fairly simple:

- Use ```render.py``` to generate photorealistic views of the textured model and produce corresponding ground truth poses and camera intrinsics files.
- Use ```vis_poses.py``` to visualize the ground truth/estimated poses.
- Throught the ```config.yaml``` or the CLI the user can control parameters such as:

    - Width, Height of images
    - Phi, theta, distance intervals
    - Background color
    - Shaders
    - Lighting color and position
    - Phong shading parameters: ambient, specular strength
    - Object color (when render geometry only)
    - Random rotations,translations and magnitude
    - Depth scale
  
Note: Only ```ply``` file models with texture (optional) are supported at the momment.

# Installation

```
git clone https://github.com/POSE-Lab/6DL-PoseGenerator
conda create -n dlpose python=3.10
cd 6DL-PoseGenerator
pip install -r requirements.txt
```

# Usage

1. Modify the ```config.yaml``` according to your needs, enter the model path and the outpath path.

2. Run ```python render.py --io.config config.yaml``` 

You can find a full list of available parameters also from the CLI by typing ```python render.py --help```.


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

## Demo example 1

For ```./demo/cube.ply``` the rendered results can be seen below.
Note that these images were produced with ```rotation_perturbation = False``` and ```rotation_translation = False```.
### Texture + Lighting
![Texture](./demo/montage_rgb.png)
### Geometry + Lighting
![Geom](./demo/montage_geom.png)
### Depth (Depth scale = 5000)
![Depth](./demo/montage_depth.png)


## Demo example 2

With ```rotation_perturbation = True``` and ```rotation_translation = True```.

![Texture_pert](./demo/montage_rgb_pert.png)