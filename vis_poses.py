from utils.io import VisualizeParams, read_json
import tyro
import glob
import os
import numpy as np
from Renderer import (
    CreateFramebuffer,
    shader_program_from_shaders,
    OPENGL_TO_OPENCV,
    init_glfw,
    save_DrawBuffer_render_rgba,
)
import cv2 as cv
from OpenGL.GL import *
from Model import Model
from Camera import camera as Camera
from os.path import join as jn
import pyrr
import tqdm


def remove_black_background(rendered_image, threshold=0):
    """
    Remove the black background from an image by creating a mask.

    Parameters:
    rendered_image (np.ndarray): The input image with a black background.
    threshold (int): The intensity threshold to identify black pixels (0-255).

    Returns:
    np.ndarray: The isolated object with transparency.
    np.ndarray: The binary mask of the object.
    """
    # Convert the image to grayscale
    gray = cv.cvtColor(rendered_image, cv.COLOR_BGR2GRAY)

    # Create a binary mask where the black areas are excluded
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    # Create an alpha channel from the mask
    alpha_channel = mask.astype(float) / 255.0

    # Add the alpha channel to the image
    isolated_object = cv.merge([rendered_image, (alpha_channel * 255).astype(np.uint8)])

    return isolated_object, alpha_channel


def overlay_with_alpha(background, object_image, alpha_mask, opacity=1.0):
    """
    Overlay an object with transparency onto a background.

    Parameters:
    background (np.ndarray): The background image.
    object_image (np.ndarray): The image of the object with an alpha channel.
    alpha_mask (np.ndarray): The alpha mask of the object (values 0-1).
    opacity (float): The opacity of the object (0 = fully transparent, 1 = fully opaque).

    Returns:
    np.ndarray: The resulting blended image.
    """
    # Resize object image and alpha mask to match the background if needed
    if background.shape[:2] != object_image.shape[:2]:
        object_image = cv.resize(
            object_image, (background.shape[1], background.shape[0])
        )
        alpha_mask = cv.resize(alpha_mask, (background.shape[1], background.shape[0]))

    # Extract RGB channels from the object image
    object_rgb = object_image[:, :, :3]

    # Blend the object with the background using the alpha mask
    blended = (
        object_rgb * (alpha_mask[..., None] * opacity)
        + background * (1 - alpha_mask[..., None] * opacity)
    ).astype(np.uint8)

    return blended


def run(args):

    poses = read_json(args.poses)
    model = Model(filename=args.model_path)
    camera_params = read_json(args.camera_params)
    image_filenames = sorted(
        glob.glob(args.images + "/*.png"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath)
    height, width, _ = cv.imread(image_filenames[0]).shape
    # create framebuffer
    init_glfw(width, height)
    framebuffer = CreateFramebuffer(width, height, GL_RGB32F, GL_RGBA, GL_FLOAT)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.ID)
    camera = Camera()

    arr_vrt_normals = np.array(model.pos_normal_buffer, dtype=np.float32)
    arr_ind = np.array(model.face_indices, dtype=np.uint32).flatten()
    stats = model.getModelStats()
    far, near = camera.getClipingPlanes(stats)

    VAO_triangles = glGenVertexArrays(1)
    VBO_triangles = glGenBuffers(1)
    EBO_triangles = glGenBuffers(1)

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
        jn(args.shaders_path, "vertex_shader.glsl"),
        None,
        jn(args.shaders_path, "fragment_shader.glsl"),
    )
    glClearColor(0.0, 0.0, 0.0, 0.0)

    for idx, img in enumerate(tqdm.tqdm(image_filenames, desc="Processing images")):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        pose_etry = poses[os.path.basename(img).strip(".png")]
        rotation = pose_etry[0]["cam_R_m2c"]
        translation = pose_etry[0]["cam_t_m2c"]
        RT = np.eye(4)
        RT[:3, :3] = np.array(rotation).reshape(3, 3)
        RT[:3, -1] = np.array(translation).reshape(
            3,
        )
        K = camera_params[os.path.basename(img).strip(".png")]["cam_K"]
        K = np.asarray(K).reshape(3, 3).astype(float)

        proj = [
            2 * K[0, 0] / float(width),
            -2 * 0 / float(width),
            0,
            0,
            0,
            2 * K[1, 1] / float(height),
            0,
            0,
            1 - 2 * (K[0, 2] / float(width)),
            2 * (K[1, 2] / float(height)) - 1,
            -(far + near) / (far - near),
            -1,
            0,
            0,
            2 * far * near / (near - far),
            0,
        ]
        proj = np.asarray(proj).reshape(4, 4)
        RT = RT.T @ OPENGL_TO_OPENCV
        res = np.dot(
            np.array([RT[0, -1], RT[1, -1], RT[2, -1], 1.0]), np.linalg.inv(RT)
        )
        model_mat = pyrr.Matrix44.identity()
        glUseProgram(shader_triangles)
        glEnable(GL_DEPTH_TEST)

        model_loc = glGetUniformLocation(shader_triangles, "model")
        view_loc = glGetUniformLocation(shader_triangles, "view")
        projection_loc = glGetUniformLocation(shader_triangles, "projection")

        objectColor = glGetUniformLocation(shader_triangles, "objectColor")
        lightColor = glGetUniformLocation(shader_triangles, "lightColor")
        lightPos = glGetUniformLocation(shader_triangles, "lightPos")
        viewPos = glGetUniformLocation(shader_triangles, "viewPos")
        ambient_strength_loc = glGetUniformLocation(shader_triangles, "ambientStrength")
        specular_strength_loc = glGetUniformLocation(
            shader_triangles, "specularStrength"
        )

        glUniform3fv(objectColor, 1, pyrr.Vector3(args.object_color))
        glUniform3fv(lightColor, 1, pyrr.Vector3(args.light_color))
        glUniform3fv(viewPos, 1, pyrr.Vector3([-RT[0, -1], -RT[1, -1], -RT[2, -1]]))
        glUniform3fv(lightPos, 1, pyrr.Vector3([res[0], res[1], res[2]]))
        glUniform1f(ambient_strength_loc, args.ambient_strength)
        glUniform1f(specular_strength_loc, args.specular_strength)

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_mat)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, RT)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, proj)
        glBindVertexArray(VAO_triangles)
        glDrawElements(GL_TRIANGLES, len(arr_ind), GL_UNSIGNED_INT, None)

        rendered_img = save_DrawBuffer_render_rgba(
            framebuffer, int(os.path.basename(img).strip(".png"))
        )
        background = cv.imread(img).astype(float)
        # Remove black background from rendered image
        isolated_object, alpha_mask = remove_black_background(rendered_img)

        # Overlay the isolated object onto the background
        overlayed = overlay_with_alpha(
            background, isolated_object, alpha_mask, opacity=args.opacity
        )
        cv.imwrite(args.outPath + f"/{idx}.png", overlayed)


if __name__ == "__main__":
    args = tyro.cli(VisualizeParams)
    run(args)
