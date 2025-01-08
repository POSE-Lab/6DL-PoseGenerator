from dataclasses import dataclass
import numpy as np
from OpenGL.GL import *
import cv2
import pyrr
import os
import png
import glfw
from Camera import camera as Camera
from Shader import Shader
from os.path import join as jn
from Model import Model
from PIL import Image
import math
import tqdm
import random
from utils.io import write_poses, write_camera_params

OPENGL_TO_OPENCV = np.array(
    [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
).reshape((4, 4))


@dataclass
class Framebuffer:
    """
    A class used to represent a Framebuffer.
    Attributes
    ----------
    ID : int
        The identifier for the framebuffer.
    colorBuffer : int
        The identifier for the color buffer associated with the framebuffer.
    width : int
        The width of the framebuffer.
    height : int
        The height of the framebuffer.
    """

    ID: int = None
    colorBuffer: int = None
    width: int = None
    height: int = None


@dataclass
class params:
    """
    Class to hold parameters for the 6DL Pose Generator Renderer.
    Attributes:
        distances (list[float]): List of distances in meters that need to be converted to millimeters.
        startPhi (int): Starting value of the phi angle.
        stopPhi (int): Stopping value of the phi angle.
        startTheta (int): Starting value of the theta angle.
        stopTheta (int): Stopping value of the theta angle.
        phiStep (int): Step value for the phi angle.
        ThetaStep (int): Step value for the theta angle.
        mode (list[str]): List of modes as strings.
    """

    distances: list[float]  # meters - has to be converted to milimeters
    startPhi: int
    stopPhi: int
    startTheta: int
    stopTheta: int
    phiStep: int
    ThetaStep: int
    mode: list[str]


def init_glfw(window_width: int, window_height: int) -> None:
    """
    Initializes the GLFW library and creates an invisible OpenGL window.
    This function initializes the GLFW library, sets the window to be invisible,
    creates an OpenGL window with the specified width and height, and makes the
    OpenGL context current. If GLFW initialization or window creation fails,
    the function terminates the GLFW library.
    Args:
        window_width (int): The width of the window to be created.
        window_height (int): The height of the window to be created.
    Returns:
        None
    """

    if not glfw.init():
        return
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(
        window_width, window_height, "Syntetic Data Generation Tool", None, None
    )
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    glfw.make_context_current(window)
    if not window:
        glfw.terminate()
        return


def CalcRenderedImagesNumber(len_dist: int, len_phi_angles: int, len_theta_angles: int):
    """
    Calculates the total number of imagse to be rendered based on the render parameters.
    Note that the output number is refering to the number of scenes rendered. (You may render
    both texture and depth. In this case the total images are imNum*2)
    """
    imNum = len_dist * len_phi_angles * len_theta_angles
    return imNum


def save_depth(path: str, im: np.ndarray) -> None:
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError("Only PNG format is currently supported.")

    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, "wb") as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def CreateFramebuffer(
    FBO_WIDTH: int, FBO_HEIGHT: int, internalFormat: int, format: int, type: int
) -> Framebuffer:
    """
    Creates a framebuffer with a color attachment and a depth-stencil renderbuffer.
    Parameters:
        FBO_WIDTH (int): The width of the framebuffer.
        FBO_HEIGHT (int): The height of the framebuffer.
        internalFormat (int): The internal format of the texture.
        format (int): The format of the texture.
        type (int): The data type of the texture.
    Returns:
        Framebuffer: An instance of the Framebuffer class containing the framebuffer ID, texture color buffer ID, width, and height.
    Raises:
        RuntimeError: If the framebuffer is not complete.
    Notes:
    - The function generates and binds a framebuffer.
    - It creates a texture to be used as the color attachment.
    - It creates a renderbuffer for depth and stencil attachment.
    - It checks if the framebuffer is complete and raises an error if not.
    - Finally, it unbinds the framebuffer and returns a Framebuffer object.
    """

    framebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    print("Framebuffer : " + str(framebuffer))
    # color attachemnt - texture
    # empty texture that will be filled by rendering to the buffer
    textureColorBuffer = glGenTextures(1)
    print("Colorbuffer: " + str(textureColorBuffer))
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer)
    glTexImage2D(
        GL_TEXTURE_2D, 0, internalFormat, FBO_WIDTH, FBO_HEIGHT, 0, format, type, None
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0
    )

    # create a renderbuffer object for depth and stencil attachment
    RBO = glGenRenderbuffers(1)
    print("RenderBuffer: " + str(RBO))
    glBindRenderbuffer(GL_RENDERBUFFER, RBO)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, FBO_WIDTH, FBO_HEIGHT)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO
    )

    # check if the framebuffer was properly created and complete
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("ERROR::FRAMEBUFFER: Framebuffer is not complete.\n")
        print(hex(glCheckFramebufferStatus(GL_FRAMEBUFFER)))
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    FRB = Framebuffer(framebuffer, textureColorBuffer, FBO_WIDTH, FBO_HEIGHT)

    return FRB


def CaptureFramebufferScene(framebuffer: Framebuffer, savePath: str) -> None:
    """
    Captures the current scene from the specified framebuffer and saves it as an image.
    This function binds the given framebuffer, reads its pixel data, processes the image
    by inverting it vertically, converts the color format from BGR to RGB, and saves the
    resulting image to the specified path.
    Args:
        framebuffer (Framebuffer): The framebuffer object containing the scene to capture.
        savePath (str): The file path where the captured image will be saved.
    Raises:
        cv2.error: If an error occurs during image processing or saving.
    Note:
        The function binds the framebuffer to GL_READ_FRAMEBUFFER for reading pixel data
        and reverts to the default framebuffer after capturing the scene.
    """

    # bind the framebuffer that will be sampled and its colorbuffer to read mode
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID)
    # glReadBuffer(framebuffer.colorBuffer)

    pixels = glReadPixels(
        0, 0, framebuffer.width, framebuffer.height, GL_RGB, GL_UNSIGNED_BYTE
    )

    # invert the image, pixel -> 1D array
    # TODO:

    # Different hundling when user requests capturing of a depth image

    try:
        img = np.frombuffer(pixels, dtype=np.uint8)

        img = np.reshape(img, (framebuffer.height, framebuffer.width, 3))
        rev_pixels = img[::-1, :]
        img = cv2.cvtColor(rev_pixels, cv2.COLOR_BGR2RGB)
        print(img.size)
        cv2.imwrite(savePath, img)
    except cv2.error as e:
        print(e)

    # bind to the deafault framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def ProjectFramebuffer(sourceFRB: Framebuffer, windowSize: tuple[int, int]) -> None:
    """
    Projects the contents of a framebuffer onto the default framebuffer (screen).
    This function binds the source framebuffer for reading and the default framebuffer for drawing,
    then blits (copies) the contents from the source framebuffer to the default framebuffer.
    Args:
        sourceFRB: The source framebuffer object containing the ID, width, and height attributes.
        windowSize (tuple): A tuple containing the width and height of the window to which the framebuffer
                            contents will be projected.
    Returns:
        None
    """

    glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFRB.ID)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

    glBlitFramebuffer(
        0,
        0,
        sourceFRB.width,
        sourceFRB.height,
        0,
        0,
        windowSize[0],
        windowSize[1],
        GL_COLOR_BUFFER_BIT,
        GL_LINEAR,
    )


def save_DrawBuffer_render_rgb(
    framebuffer: Framebuffer, img_id: int, savePath=None, subfolder_name=None
):
    """
    Save or return an RGB image rendered from the OpenGL framebuffer.
    This function reads pixel data from the OpenGL framebuffer, processes it to
    create an RGB image, and either saves the image to a specified path or returns
    it directly.
    Args:
        framebuffer: The OpenGL framebuffer object containing the rendered image.
        img_id (int): The identifier for the image, used in the filename if saving.
        savePath (str, optional): The base directory where the image will be saved.
                                  If None, the image is returned instead of saved.
        subfolder_name (str, optional): The subfolder name within the savePath where
                                        the image will be saved. If None, no subfolder
                                        is created.
    Returns:
        numpy.ndarray: The processed RGB image if savePath is None.
    Raises:
        cv2.error: If an error occurs during image processing or saving.
    Notes:
        - The image is read from the framebuffer in BGR format and converted to RGB.
        - The image is resized to the dimensions of the framebuffer.
        - If savePath is provided, the image is saved in the specified directory
          structure: savePath/subfolder_name/rgb/img_id.png.
    """

    pixels = glReadPixels(
        0, 0, framebuffer.width, framebuffer.height, GL_RGB, GL_UNSIGNED_BYTE
    )

    try:
        img = np.frombuffer(pixels, dtype=np.uint8)

        img = np.reshape(img, (framebuffer.height, framebuffer.width, 3))
        rev_pixels = img[::-1, :]
        img = cv2.cvtColor(rev_pixels, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (framebuffer.width, framebuffer.height))
        if savePath is not None:
            if not os.path.exists(savePath + f"/{subfolder_name}"):
                os.makedirs(savePath + f"/{subfolder_name}")
            if not os.path.exists(savePath + f"/{subfolder_name}/rgb"):
                os.makedirs(savePath + f"/{subfolder_name}/rgb")
            cv2.imwrite(
                savePath + f"/{subfolder_name}/rgb" + "/" + str(img_id) + ".png", img
            )
        else:
            return img
    except cv2.error as e:
        print(e)


def save_DrawBuffer_render_rgba(
    framebuffer: Framebuffer,
    img_id: int,
    savePath: str = None,
    subfolder_name: str = None,
):
    """
    Save the rendered RGBA buffer to an image file or return it as a numpy array.
    This function reads the pixel data from the OpenGL framebuffer, processes it,
    and either saves it as an image file or returns it as a numpy array.
    Args:
        framebuffer: The OpenGL framebuffer object containing the rendered image.
        img_id (int): The identifier for the image, used in the filename if saving.
        savePath (str, optional): The directory path where the image will be saved.
                                  If None, the image will be returned as a numpy array.
        subfolder_name (str, optional): The name of the subfolder within savePath
                                        where the image will be saved. If None, no subfolder is created.
    Returns:
        numpy.ndarray: The processed image as a numpy array if savePath is None.
    Raises:
        cv2.error: If an error occurs during image processing or saving.
    """

    pixels = glReadPixels(
        0, 0, framebuffer.width, framebuffer.height, GL_RGBA, GL_UNSIGNED_BYTE
    )

    try:
        img = np.frombuffer(pixels, dtype=np.uint8)

        img = np.reshape(img, (framebuffer.height, framebuffer.width, 4))
        rev_pixels = img[::-1, :]
        img = cv2.cvtColor(rev_pixels, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (framebuffer.width, framebuffer.height))
        if savePath is not None:
            if not os.path.exists(savePath + f"/{subfolder_name}"):
                os.makedirs(savePath + f"/{subfolder_name}")
            if not os.path.exists(savePath + f"/{subfolder_name}/rgb"):
                os.makedirs(savePath + f"/{subfolder_name}/rgb")
            cv2.imwrite(
                savePath + f"/{subfolder_name}/rgb" + "/" + str(img_id) + ".png", img
            )
        else:
            return img
    except cv2.error as e:
        print(e)


def save_depth(path: str, im: np.ndarray, framebuffer: Framebuffer) -> None:
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError("Only PNG format is currently supported.")

    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)
    im_uint16 = cv2.resize(im_uint16, (int(framebuffer.width), int(framebuffer.height)))

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(
        im_uint16.shape[1], im_uint16.shape[0], greyscale=True, bitdepth=16
    )
    with open(path, "wb") as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im_uint16.shape[1])))


def save_DrawBuffer_render_depth(framebuffer: Framebuffer, img_id: int, savePath: str, subfolder_name: str) -> None:
    """
    Saves the depth buffer rendered from the given framebuffer to a specified path.
    Args:
        framebuffer: The framebuffer object containing the rendered image.
        img_id (int): The identifier for the image to be saved.
        savePath (str): The base directory where the image will be saved.
        subfolder_name (str): The subfolder name within the base directory where the image will be saved.
    Raises:
        cv2.error: If an error occurs during the saving process.
    Description:
        This function reads the pixel data from the given framebuffer, processes it to extract the depth information,
        and saves it as a PNG image in the specified directory. The image is saved in a subfolder named `subfolder_name`
        within the `savePath` directory, with the filename being the `img_id` followed by the ".png" extension.
    """

    img_from_buffer = np.zeros(
        (framebuffer.height, framebuffer.width, 3), dtype=np.float32
    )

    glReadPixels(
        0, 0, framebuffer.width, framebuffer.height, GL_RGB, GL_FLOAT, img_from_buffer
    )

    try:
        img_from_buffer.shape = (framebuffer.height, framebuffer.width, 3)
        img_from_buffer = img_from_buffer[::-1, :]
        img_from_buffer = img_from_buffer[:, :, 0]

        if not os.path.exists(savePath + f"/{subfolder_name}"):
            os.makedirs(savePath + f"/{subfolder_name}")

        save_depth(
            savePath + f"/{subfolder_name}" + "/" + str(img_id) + ".png",
            img_from_buffer,
            framebuffer,
        )
    except cv2.error as e:
        print(e)


def shader_program_from_shaders(
    vertex_shader_file: str, geometry_shader_file: str, fragment_shader_file: str
) -> None:
    """
    Creates a shader program from given vertex, geometry, and fragment shader files.
    This function reads the shader source code from the provided files, compiles the shaders,
    and links them into a shader program.
    Args:
        vertex_shader_file (str): Path to the vertex shader file.
        geometry_shader_file (str): Path to the geometry shader file.
        fragment_shader_file (str): Path to the fragment shader file.
    Returns:
        int: The handle of the created shader program.
    """

    shaders = Shader(vertex_shader_file, fragment_shader_file, geometry_shader_file)
    shaders.readShadersFromFile()
    vertex, fragment, geometry = shaders.compileShader()
    shader_program = shaders.compileProgram(vertex, fragment, geometry)

    return shader_program


def rotateY(angle: float) -> np.ndarray:
    """
    Generates a 3x3 rotation matrix for a rotation around the Y-axis.
    Parameters:
    angle (float): The angle in degrees to rotate around the Y-axis.
    Returns:
    numpy.ndarray: A 3x3 rotation matrix representing the rotation around the Y-axis.
    """

    angle = deg2rad(angle)
    rot = np.array(
        [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]
    )
    return rot


def rotateX(angle: float) -> np.ndarray:
    """
    Generates a rotation matrix for a rotation around the X-axis.
    Parameters:
    angle (float): The angle in degrees by which to rotate around the X-axis.
    Returns:
    np.ndarray: A 3x3 rotation matrix representing the rotation around the X-axis.
    """

    angle = deg2rad(angle)
    rot = np.array(
        [
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ]
    )
    return rot


def rotateZ(angle: float) -> np.ndarray:
    """
    Generate a rotation matrix for a rotation around the Z-axis.
    Parameters:
    angle (float): The angle in degrees by which to rotate.
    Returns:
    numpy.ndarray: A 3x3 rotation matrix representing the rotation around the Z-axis.
    """

    angle = deg2rad(angle)
    rot = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return rot


def deg2rad(deg: float) -> float:
    """
    Convert degrees to radians.
    Parameters:
    deg (float): Angle in degrees.
    Returns:
    float: Angle in radians.
    """

    return (deg / 180.0) * math.pi


def compute_K_from_perspective(fov: float, width: int, height: int) -> np.ndarray:
    """
    Compute the intrinsic camera matrix (K) from the given perspective parameters.
    Parameters:
        fov (float): Field of view in degrees.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
    Returns:
        numpy.ndarray: The intrinsic camera matrix (K) of shape (3, 3).
    Description:
        This function calculates the intrinsic camera matrix based on the provided field of view (fov),
        image width, and image height. The focal length is computed using the vertical field of view,
        and the principal point is assumed to be at the center of the image. The resulting matrix K
        is used in computer vision and graphics to describe the camera's internal parameters.
    """

    fov_rad = np.radians(fov)

    # Focal length in pixels (using vertical FOV)
    f = height / (2 * np.tan(fov_rad / 2))

    # Principal point coordinates (center of the image)
    c_x = width / 2
    c_y = height / 2

    # Intrinsic matrix K
    K = np.array([[f, 0, c_x], [0, f, c_y], [0, 0, 1]])
    return K


def RenderRGBD(params: params):
    """
    RenderRGBD renders RGBD images based on the provided parameters.
    Parameters:
        params (params): A parameter object containing the following attributes:
            - FBO_WIDTH (int): Framebuffer width.
            - FBO_HEIGHT (int): Framebuffer height.
            - model_path (str): Path to the 3D model file.
            - render_modes (list): List of rendering modes ('lines', 'texture', 'depth', 'triangles').
            - shaders_path (str): Path to the shaders directory.
            - texture_file (str): Path to the texture file.
            - distances (tuple): Tuple containing start, end, and step values for distances.
            - phis (tuple): Tuple containing start, end, and step values for phi angles.
            - thetas (tuple): Tuple containing start, end, and step values for theta angles.
            - background_color (tuple): Background color for the rendering.
            - rotation_perturbation (bool): Flag to enable/disable rotation perturbation.
            - rotation_xrange (tuple): Range for x-axis rotation perturbation.
            - rotation_yrange (tuple): Range for y-axis rotation perturbation.
            - rotation_zrange (tuple): Range for z-axis rotation perturbation.
            - translation_perturbation (bool): Flag to enable/disable translation perturbation.
            - translation_xrange (tuple): Range for x-axis translation perturbation.
            - translation_yrange (tuple): Range for y-axis translation perturbation.
            - translation_zrange (tuple): Range for z-axis translation perturbation.
            - triangles_object_color (tuple): Color of the object in triangles mode.
            - triangles_light_color (tuple): Color of the light in triangles mode.
            - triangles_ambient_strength (float): Ambient strength in triangles mode.
            - triangles_specular_strength (float): Specular strength in triangles mode.
            - texture_light_color (tuple): Color of the light in texture mode.
            - texture_ambient_strength (float): Ambient strength in texture mode.
            - texture_specular_strengh (float): Specular strength in texture mode.
            - light_position (tuple): Position of the light source.
            - depth_scale (float): Scale for depth rendering.
            - savePath (str): Path to save the rendered images.
            - object_id (int): ID of the object being rendered.
    Returns:
        None
    Description:
        This function initializes the framebuffer, loads the 3D model, sets up the camera,
        and renders images in different modes (lines, texture, depth, triangles) based on
        the provided parameters. It also applies random pose perturbations if specified and
        saves the rendered images and camera parameters to the specified path.
    """

    framebuffer = CreateFramebuffer(
        params.FBO_WIDTH, params.FBO_HEIGHT, GL_RGB32F, GL_RGB, GL_FLOAT
    )
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.ID)
    camera = Camera()

    print(f"Loading model {params.model_path}...")
    model = Model(filename=params.model_path)
    stats = model.getModelStats()
    far, near = camera.getClipingPlanes(stats)
    print(f"Far: {far}, Near: {near}")

    arr_vrt = model.model_points.flatten()
    arr_vrt_normals = np.array(model.pos_normal_buffer, dtype=np.float32)
    arr_ind = np.array(model.face_indices, dtype=np.uint32).flatten()

    if "lines" in params.render_modes:
        arr_ind_lines = np.array(model.lines_indices, dtype=np.uint32)
    if "texture" in params.render_modes:
        # print(model.uvs)
        arr_texture = np.array(model.pose_normal_uvs_buffer, dtype=np.float32)

    if "depth" in params.render_modes:
        VAO_depth = glGenVertexArrays(1)
        VBO_depth = glGenBuffers(1)
        EBO_depth = glGenBuffers(1)

        glBindVertexArray(VAO_depth)
        glBindBuffer(GL_ARRAY_BUFFER, VBO_depth)
        glBufferData(GL_ARRAY_BUFFER, arr_vrt.nbytes, arr_vrt, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_depth)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, arr_ind.nbytes, arr_ind, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        shader_depth = shader_program_from_shaders(
            jn(params.shaders_path, "vertex_shader_depth.glsl"),
            None,
            jn(params.shaders_path, "fragment_shader_depth.glsl"),
        )

    if "triangles" in params.render_modes:
        VAO_triangles = glGenVertexArrays(1)
        VBO_triangles = glGenBuffers(1)
        EBO_triangles = glGenBuffers(1)
        arr_texture = np.array(model.pose_normal_uvs_buffer, dtype=np.float32)
        glBindVertexArray(VAO_triangles)
        glBindBuffer(GL_ARRAY_BUFFER, VBO_triangles)
        glBufferData(
            GL_ARRAY_BUFFER, arr_vrt_normals.nbytes, arr_vrt_normals, GL_STATIC_DRAW
        )

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_triangles)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, arr_ind.nbytes, arr_ind, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, arr_vrt_normals.itemsize * 6, ctypes.c_void_p(0)
        )

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, arr_vrt_normals.itemsize * 6, ctypes.c_void_p(12)
        )
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        shader_triangles = shader_program_from_shaders(
            jn(params.shaders_path, "vertex_shader.glsl"),
            None,
            jn(params.shaders_path, "fragment_shader.glsl"),
        )

    if "lines" in params.render_modes:
        VAO_lines = glGenVertexArrays(1)
        VBO_lines = glGenBuffers(1)
        EBO_lines = glGenBuffers(1)

        # Create buffer for lines render
        glBindVertexArray(VAO_lines)
        glBindBuffer(GL_ARRAY_BUFFER, VBO_lines)
        glBufferData(GL_ARRAY_BUFFER, arr_vrt.nbytes, arr_vrt, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_lines)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, arr_ind_lines.nbytes, arr_ind_lines, GL_STATIC_DRAW
        )

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        shader_lines = shader_program_from_shaders(
            jn(params.shaders_path, "vertex_shader.glsl"),
            None,
            jn(params.shaders_path, "fragment_shader.glsl"),
        )

    if "texture" in params.render_modes:
        VAO_texture = glGenVertexArrays(1)
        VBO_texture = glGenBuffers(1)
        EBO_triangles_texture = glGenBuffers(1)

        glBindVertexArray(VAO_texture)
        glBindBuffer(GL_ARRAY_BUFFER, VBO_texture)
        glBufferData(GL_ARRAY_BUFFER, arr_texture.nbytes, arr_texture, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_triangles_texture)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, arr_ind.nbytes, arr_ind, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, arr_texture.itemsize * 8, ctypes.c_void_p(0)
        )

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, arr_texture.itemsize * 8, ctypes.c_void_p(12)
        )

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(
            2, 2, GL_FLOAT, GL_FALSE, arr_texture.itemsize * 8, ctypes.c_void_p(24)
        )

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image_tex = Image.open(params.texture_file)  # "FINAL_8192_edited.png"
        tex_image_data = image_tex.convert("RGBA").tobytes()
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            image_tex.width,
            image_tex.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            tex_image_data,
        )

        shader_texture = shader_program_from_shaders(
            jn(params.shaders_path, "vertex_shader_texture.glsl"),
            None,
            jn(params.shaders_path, "fragment_shader_texture.glsl"),
        )

    dists = range(
        params.distances[0],
        params.distances[1] + params.distances[2],
        params.distances[2],
    )
    phis = range(params.phis[0], params.phis[1] + params.phis[2], params.phis[2])
    thetas = range(
        params.thetas[0], params.thetas[1] + params.thetas[2], params.thetas[2]
    )
    print(len(dists), len(thetas), len(phis))
    progress = 0
    num_items_rendered = CalcRenderedImagesNumber(len(dists), len(thetas), len(phis))
    Rs, Ts = [], []
    print(f"#{num_items_rendered} items will be rendered")
    glClearColor(*params.background_color)
    with tqdm.tqdm(range(num_items_rendered), desc="Rendering...") as pbar:
        for dist in dists:
            for theta in thetas:
                for phi in phis:

                    camera.setParams(phi, theta, dist)
                    camera.update_camera_parameters()

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                    model_matrix = pyrr.Matrix44.identity()
                    projection = pyrr.Matrix44.perspective_projection(
                        45,
                        float(params.FBO_WIDTH) / float(params.FBO_HEIGHT),
                        near,
                        far,
                    )

                    K = compute_K_from_perspective(
                        45, params.FBO_WIDTH, params.FBO_HEIGHT
                    )
                    # print("Intrinsic Matrix K:", K)

                    # random pose perturbation
                    if params.rotation_perturbation:
                        r_x = random.uniform(*params.rotation_xrange)
                        r_y = random.uniform(*params.rotation_yrange)
                        r_z = random.uniform(*params.rotation_zrange)

                        camera.tranformation *= (
                            pyrr.Matrix44.from_matrix33(rotateX(r_x))
                            * pyrr.Matrix44.from_matrix33(rotateY(r_y))
                            * pyrr.Matrix44.from_matrix33(rotateZ(r_z))
                        )

                    if params.translation_perturbation:

                        x_n = random.uniform(*params.translation_xrange)
                        y_n = random.uniform(*params.translation_yrange)
                        y_n = random.uniform(*params.translation_zrange)

                        camera.tranformation[3, 0] += x_n
                        camera.tranformation[3, 1] += y_n
                        camera.tranformation[3, 2] += y_n

                    RT = OPENGL_TO_OPENCV @ np.transpose(camera.tranformation)
                    Rs.append(RT[0:3, 0:3].flatten())
                    Ts.append(RT[0:3, 3].flatten())

                    if "triangles" in params.render_modes:
                        glUseProgram(shader_triangles)
                        glEnable(GL_DEPTH_TEST)
                        model_ = glGetUniformLocation(shader_triangles, "model")
                        view_ = glGetUniformLocation(shader_triangles, "view")
                        projection_ = glGetUniformLocation(
                            shader_triangles, "projection"
                        )

                        objectColor = glGetUniformLocation(
                            shader_triangles, "objectColor"
                        )
                        lightColor = glGetUniformLocation(
                            shader_triangles, "lightColor"
                        )
                        lightPos = glGetUniformLocation(shader_triangles, "lightPos")
                        viewPos = glGetUniformLocation(shader_triangles, "viewPos")
                        ambient_strength_loc = glGetUniformLocation(
                            shader_triangles, "ambientStrength"
                        )
                        specular_strength_loc = glGetUniformLocation(
                            shader_triangles, "specularStrength"
                        )

                        glUniform3fv(
                            objectColor, 1, pyrr.Vector3(params.triangles_object_color)
                        )
                        glUniform3fv(
                            lightColor, 1, pyrr.Vector3(params.triangles_light_color)
                        )
                        glUniform3fv(
                            viewPos,
                            1,
                            pyrr.Vector3(
                                [
                                    camera.cameraPos[0],
                                    camera.cameraPos[1],
                                    camera.cameraPos[2],
                                ]
                            ),
                        )
                        glUniform3fv(
                            lightPos,
                            1,
                            pyrr.Vector3(
                                [
                                    camera.cameraPos[0],
                                    camera.cameraPos[1],
                                    camera.cameraPos[2],
                                ]
                                if params.light_position is None
                                else params.light_position
                            ),
                        )
                        glUniform1f(
                            ambient_strength_loc, params.triangles_ambient_strength
                        )
                        glUniform1f(
                            specular_strength_loc, params.triangles_specular_strength
                        )

                        glUniformMatrix4fv(model_, 1, GL_FALSE, model_matrix)
                        glUniformMatrix4fv(view_, 1, GL_FALSE, camera.tranformation)
                        glUniformMatrix4fv(projection_, 1, GL_FALSE, projection)
                        glBindVertexArray(VAO_triangles)
                        glDrawElements(
                            GL_TRIANGLES, len(arr_ind), GL_UNSIGNED_INT, None
                        )

                        save_DrawBuffer_render_rgb(
                            framebuffer,
                            progress,
                            params.savePath,
                            subfolder_name="geom",
                        )

                    if "texture" in params.render_modes:
                        glBindTexture(GL_TEXTURE_2D, texture)
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                        glUseProgram(shader_texture)
                        glEnable(GL_DEPTH_TEST)

                        # same trasnformation for all modes
                        model_ = glGetUniformLocation(shader_texture, "model")
                        view_ = glGetUniformLocation(shader_texture, "view")
                        projection_ = glGetUniformLocation(shader_texture, "projection")

                        lightColor = glGetUniformLocation(shader_texture, "lightColor")
                        lightPos = glGetUniformLocation(shader_texture, "lightPos")
                        viewPos = glGetUniformLocation(shader_texture, "viewPos")
                        ambient_strength_loc = glGetUniformLocation(
                            shader_texture, "ambientStrength"
                        )
                        specular_strength_loc = glGetUniformLocation(
                            shader_texture, "specularStrength"
                        )

                        glUniform3fv(
                            lightColor, 1, pyrr.Vector3(params.texture_light_color)
                        )
                        glUniform3fv(
                            viewPos,
                            1,
                            pyrr.Vector3(
                                [
                                    camera.cameraPos[0],
                                    camera.cameraPos[1],
                                    camera.cameraPos[2],
                                ]
                            ),
                        )
                        glUniform3fv(
                            lightPos,
                            1,
                            pyrr.Vector3(
                                [
                                    camera.cameraPos[0],
                                    camera.cameraPos[1],
                                    camera.cameraPos[2],
                                ]
                                if params.light_position is None
                                else params.light_position
                            ),
                        )

                        # model_matrix = pyrr.Matrix44.from_matrix33(rotateY(90.))
                        glUniformMatrix4fv(model_, 1, GL_FALSE, model_matrix)
                        glUniformMatrix4fv(view_, 1, GL_FALSE, camera.tranformation)
                        glUniformMatrix4fv(projection_, 1, GL_FALSE, projection)

                        glUniform1f(
                            ambient_strength_loc, params.texture_ambient_strength
                        )
                        glUniform1f(
                            specular_strength_loc, params.texture_specular_strengh
                        )

                        glBindVertexArray(VAO_texture)
                        glDrawElements(
                            GL_TRIANGLES, len(arr_ind), GL_UNSIGNED_INT, None
                        )

                        save_DrawBuffer_render_rgb(
                            framebuffer,
                            progress,
                            params.savePath,
                            subfolder_name="texture",
                        )

                    if "depth" in params.render_modes:
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                        glEnable(GL_DEPTH_TEST)
                        glDisable(GL_CULL_FACE)

                        glUseProgram(shader_depth)

                        model_depth_ = glGetUniformLocation(shader_depth, "model")
                        projectionLoc = glGetUniformLocation(shader_depth, "projection")
                        viewLoc = glGetUniformLocation(shader_depth, "view")
                        depth_scale_loc = glGetUniformLocation(
                            shader_depth, "depth_scale"
                        )
                        glUniformMatrix4fv(model_depth_, 1, GL_FALSE, model_matrix)
                        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, camera.tranformation)
                        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection)
                        glUniform1f(depth_scale_loc, params.depth_scale)

                        # bind depth VAO
                        glBindVertexArray(VAO_depth)
                        glDrawElements(
                            GL_TRIANGLES, len(arr_ind), GL_UNSIGNED_INT, None
                        )

                        save_DrawBuffer_render_depth(
                            framebuffer,
                            progress,
                            params.savePath,
                            subfolder_name="depth",
                        )

                    pbar.update(1)
                    progress += 1

    print("Done rendering.")
    print(f"Writing poses to {params.savePath}/scene_gt.json...")
    write_poses(params.savePath, np.array(Rs), np.array(Ts), params.object_id)
    print(f"Writing camera parameters to {params.savePath}/scene_camera.json")
    write_camera_params(params.savePath, K, params.depth_scale, progress)
