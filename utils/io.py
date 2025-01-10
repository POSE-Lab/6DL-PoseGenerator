import os
import yaml
from .JsonEncoder import NoIndent, NoIndentEncoder
import json
from dataclasses import dataclass, field
from numpy.typing import NDArray
from .params import RenderParams


@dataclass
class VisualizeParams:

    # outpath to save visualizations
    outPath: str

    # 3d Model path
    model_path: str

    # scene_gt.json poses file
    poses: str = "./scene_gt.json"

    # opacity of the rendered model
    opacity: float = 0.7

    # image foler
    images: str = "./images"

    # instrinics file
    camera_params: str = "scene_camera.json"

    # Shaders path
    shaders_path: str = "./shaders"

    # Object color
    object_color: list = field(default_factory=lambda: [1.0, 0.5, 0.31])

    # light color
    light_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # ambient strength
    ambient_strength: float = 0.1

    # specular_strength
    specular_strength: float = 0.55


@dataclass
class RenderingParams:

    # Frame buffer object's width
    FBO_WIDTH: int = 1920
    # Frame buffer object's height
    FBO_HEIGHT: int = 1080
    # SSAA factor for anti-aliasing
    SSAA_factor: int = 4

    # Phi angles: start,stop,step
    phis: list = field(default_factory=lambda: [0, 360, 40])
    # Theta angles: start,stop,step
    thetas: list = field(default_factory=lambda: [0, 90, 30])
    # Distances: start,stop,step
    distances: list = field(default_factory=lambda: [300, 900, 100])
    # Background color
    background_color: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ShadersParams:

    # Path to shaders used for the rendering
    shaders_path: str = "./shaders/"
    # In which mode to render the object
    render_modes: list = field(default_factory=lambda: ["depth", "triangles"])
    # Path to texture file
    texture_file: str = "./texture.png"

    # Light position in the rendered scene
    light_positon: list = field(default_factory=lambda: [10.0, 100.0, 300.0])

    # scale of of the depth rendering. Real depth is RD= D/depth_scale
    depth_scale: float = 10


@dataclass
class TextureLightingParams:

    # ambient strength factor for texture rendering
    texture_ambient_strength: float = 0.3
    # specular strength factor for texture rendering
    texture_specular_strength: float = 0.55
    # color of the emitted light
    texture_light_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class GeometryLightingParams:

    # ambient strength factor for geometry rendering
    triangles_ambient_strength: float = 0.1
    # specular strength factor for geometry rendering
    triangles_specular_strength: float = 0.55
    # color of the triangle faces
    triangles_object_color: list = field(default_factory=lambda: [0.7, 0.7, 0.7])
    # color of the emitted light
    triangles_light_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class PosesParams:

    # wheter to apply random rotations
    rotation_perturbation: bool = True
    # whether to apply random translations
    translation_perturbation: bool = True
    # rotation about x axis min,max limits of uniform distribution
    rotation_xrange: list = field(default_factory=lambda: [-20.5, 10.5])
    # rotation about y axis min,max limits of uniform distribution
    rotation_yrange: list = field(default_factory=lambda: [-1.5, 1.5])
    # rotation about z axis min,max limits of uniform distribution
    rotation_zrange: list = field(default_factory=lambda: [-1.5, 1.5])
    # translation in x axis min,max limits of uniform distribution
    translation_xrange: list = field(default_factory=lambda: [-200.0, 200.0])
    # translation in y axis min,max limits of uniform distribution
    translation_yrange: list = field(default_factory=lambda: [-100.0, 100.0])
    # translation in z axis min,max limits of uniform distribution
    translation_zrange: list = field(default_factory=lambda: [-50.0, 50.0])


@dataclass
class ModelParams:
    # path to 3D model
    model_path: str = "./model3d.ply"
    # Object id for writing the camera poses
    object_id: int = 3


@dataclass
class IOParams:

    # Configuration file for rendering parameters
    config: str = "./config.yaml"
    # Path to save results
    savePath: str = "./output"


@dataclass
class Render_args:

    io: IOParams = field(default_factory=IOParams)
    rendering: RenderingParams = field(default_factory=RenderingParams)
    shaders: ShadersParams = field(default_factory=ShadersParams)
    texture_render_settings: TextureLightingParams = field(
        default_factory=TextureLightingParams
    )
    geometry_render_settings: GeometryLightingParams = field(
        default_factory=GeometryLightingParams
    )
    poses: PosesParams = field(default_factory=PosesParams)
    model: ModelParams = field(default_factory=ModelParams)


def merge_params(config_params, cli_params):
    """
    Merge CLI parameters with config parameters, giving precedence to CLI parameters.
    """
    config_dict = config_params.__dict__
    for key, value in cli_params.items():
        if value is not None:
            config_dict[key] = value
    return RenderParams(config_dict)


def parse_yaml(config_file: str) -> dict:
    """
    Parses a YAML configuration file and returns its contents as a dictionary.
    Args:
        config_file (str): The path to the YAML configuration file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
    Example:
        config = parse_yaml('config.yaml')
    """

    print(f"Parsing configuration file: {config_file}...")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist.")

    with open(config_file, "r") as f:
        data = yaml.full_load(f)

    return data


def write_poses(file: str, R: list, T: list, objid: int) -> None:
    """
    Writes pose data to a JSON file.
    Parameters:
        file (str): The directory path where the JSON file will be saved.
        R (list): A list of 3x3 rotation matrices.
        T (list): A list of 3x1 translation vectors.
        objid (int): The object ID to be included in the JSON data.

    The function creates a JSON file named 'scene_gt.json' in the specified directory.
    Each entry in the JSON file contains the rotation matrix, translation vector, and object ID.
    """

    data = {}
    with open(os.path.join(file, "scene_gt.json"), "w") as of:
        for i in range(len(R)):

            data[str(i)] = NoIndent(
                [
                    {
                        "cam_R_m2c": [
                            R[i][0],
                            R[i][1],
                            R[i][2],
                            R[i][3],
                            R[i][4],
                            R[i][5],
                            R[i][6],
                            R[i][7],
                            R[i][8],
                        ],
                        "cam_t_m2c": [
                            T[i][0],
                            T[i][1],
                            T[i][2],
                        ],  # swap y,z axis by reversing the order of t[i][1]mt[i][2]\
                        "obj_id": objid,
                    }
                ]
            )

        of.write(json.dumps(data, cls=NoIndentEncoder, indent=2))


def read_json(file: str) -> dict:
    """
    Reads a JSON file and returns the data as a Python dictionary.
    Args:
        file (str): The path to the JSON file to be read.
    Returns:
        dict: The data from the JSON file as a dictionary.
    """

    with open(file, "r") as f:
        data = json.load(f)

    return data


def write_camera_params(
    savePath: str, K: NDArray, depthscale: float, total_images: int
) -> None:
    """
    Writes camera parameters to a JSON file.
    Parameters:
        savePath (str): The directory path where the JSON file will be saved.
        K (numpy.ndarray): The camera intrinsic matrix.
        depthscale (float): The depth scale factor.
        total_images (int): The total number of images for which camera parameters are to be written.
    Returns:
        None
    """

    data = {}

    with open(os.path.join(savePath, "scene_camera.json"), "w") as of:
        for i in range(total_images):
            data[str(i)] = NoIndent(
                {"cam_K": [*K.flatten()], "depth_scale": depthscale}
            )

        of.write(json.dumps(data, cls=NoIndentEncoder, indent=2))
