"""
This class contains the following methods:
    - ReadShaderFromFile(fielanamem,type)
"""

import os
import glfw
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGL.arrays import *
from OpenGL.GL import shaders
from OpenGLContext.arrays import *


class Shader:
    """
    A class to handle the creation, compilation, and usage of OpenGL shaders.
    Attributes:
    ----------
    vertexfilename : str
        The filename of the vertex shader.
    fragmentfilename : str
        The filename of the fragment shader.
    geometryFilename : str, optional
        The filename of the geometry shader (default is None).
    Methods:
    -------
    __init__(vertexfilename, fragmentfilename, geometry_filename=None)
        Initializes the Shader object with the given shader filenames.
    printID()
        Prints the filename of the shader (currently prints an undefined attribute 'filename').
    readShadersFromFile()
        Reads the shader source code from the provided files and stores them as strings.
    compileShader()
        Compiles the vertex, fragment, and optionally geometry shaders from the source code strings.
    compileProgram(vertex_shader, fragment_shader, geometry_shader)
        Links the compiled shaders into an OpenGL shader program.
    UseProgram(programmID)
        Uses the specified OpenGL shader program.
    """

    def __init__(
        self, vertexfilename: str, fragmentfilename: str, geometry_filename: str = None
    ) -> None:
        self.vertexfilename = vertexfilename
        self.fragmentfilename = fragmentfilename
        self.geometryFilename = geometry_filename

    def printID(self) -> None:
        """
        Prints the filename attribute of the instance.
        This method outputs the value of the `filename` attribute to the console.
        """
        print(self.filename)

    def readShadersFromFile(self) -> None:
        """
        Reads shader source code from files and stores them in instance variables.
        This method reads the vertex, fragment, and optionally geometry shader source code
        from their respective files and stores the contents in the instance variables
        `vertexstring`, `fragmentstring`, and `geometrystring`.

        Instance Variables:
            - vertexfilename (str): The file path to the vertex shader source code.
            - fragmentfilename (str): The file path to the fragment shader source code.
            - geometryFilename (str, optional): The file path to the geometry shader source code.
        Raises:
            - FileNotFoundError: If any of the specified shader files do not exist.
            - IOError: If there is an error reading any of the shader files.
        """

        with open(self.vertexfilename, "r") as file:
            shader = file.read()
            self.vertexstring = shader
        with open(self.fragmentfilename, "r") as file:
            shader = file.read()
            self.fragmentstring = shader
        if self.geometryFilename is not None:
            with open(self.geometryFilename, "r") as file:
                shader = file.read()
                self.geometrystring = shader

    def compileShader(self) -> tuple[int, int, int]:
        """
        Compiles the vertex, fragment, and optionally geometry shaders.
        This method compiles the vertex and fragment shaders from the provided
        shader strings. If a geometry shader filename is provided, it also compiles
        the geometry shader.

        Returns:
            tuple: A tuple containing the compiled vertex and fragment shaders. If a
                   geometry shader is provided, the tuple will also contain the compiled
                   geometry shader; otherwise, the third element will be None.
        """

        vertex = shaders.compileShader(self.vertexstring, GL_VERTEX_SHADER)
        fragment = shaders.compileShader(self.fragmentstring, GL_FRAGMENT_SHADER)
        if self.geometryFilename is not None:
            geometry = shaders.compileShader(self.geometrystring, GL_GEOMETRY_SHADER)
            return vertex, fragment, geometry
        else:
            return vertex, fragment, None

    def compileProgram(
        self, vertex_shader: int, fragment_shader: int, geometry_shader: int
    ) -> int:
        """
        Compiles a shader program using the provided vertex, fragment, and optionally geometry shaders.
        Args:
            vertex_shader: The vertex shader source code or compiled object.
            fragment_shader: The fragment shader source code or compiled object.
            geometry_shader: The geometry shader source code or compiled object (optional).
        Returns:
            The compiled shader program.
        If the geometry shader is not provided, the program will be compiled using only the vertex and fragment shaders.
        """

        if self.geometryFilename is not None:
            program = shaders.compileProgram(
                vertex_shader, fragment_shader, geometry_shader
            )
        else:
            program = shaders.compileProgram(vertex_shader, fragment_shader)

        return program

    def UseProgram(self, programmID: int) -> None:
        """
        Activates the specified shader program.
        This method sets the current shader program to the one identified by the given program ID.
        Args:
            programmID (int): The ID of the shader program to be used.
        """

        shaders.glUseProgram(programmID)
