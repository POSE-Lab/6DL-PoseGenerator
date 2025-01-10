import open3d as o3d
import numpy as np
from plyfile import PlyData
from typing import Union


class Model:
    """
    A class to represent a 3D model loaded from a file.
    Attributes
    ----------
    filename : str
        The path to the file containing the 3D model.
    model : open3d.geometry.TriangleMesh
        The loaded 3D model as an open3d TriangleMesh object.
    model_points : np.ndarray
        The vertices of the model as a NumPy array.
    vertices : np.ndarray
        The flattened vertices of the model.
    face_indices : np.ndarray
        The flattened face indices of the model.
    normals : np.ndarray
        The vertex normals of the model.
    uvs : np.ndarray or None
        The UV coordinates per vertex if available, otherwise None.
    pos_buffer : np.ndarray
        The buffer containing vertex positions.
    pos_normal_buffer : np.ndarray
        The buffer containing interleaved vertex positions and normals.
    pose_normal_uvs_buffer : np.ndarray
        The buffer containing interleaved vertex positions, normals, and UVs if available.
    Methods
    -------
    __init__(filename: str, auto_compute_normals: bool = True) -> None
        Initializes the Model object by loading the 3D model from the specified file.
    load_per_vertex_uvs() -> np.ndarray or None
        Loads UV coordinates per vertex from the PLY file.
    create_buffers() -> None
        Creates buffers for vertex positions, normals, and UVs.
    getModelStats() -> dict
        Computes and returns statistics about the model's vertices.
    """

    def __init__(self, filename: str, auto_compute_normals: bool = True) -> None:
        self.filename = filename
        self.model = o3d.io.read_triangle_mesh(self.filename)
        self.model_points = np.asarray(self.model.vertices, dtype=np.float32)
        self.vertices = self.model_points.flatten()
        self.face_indices = np.asarray(self.model.triangles).flatten()

        if not self.model.has_vertex_normals():
            self.model.compute_vertex_normals()
        self.normals = np.asarray(self.model.vertex_normals)

        self.uvs = self.load_per_vertex_uvs()
        if self.model.has_vertex_colors():
            self.colors = np.asarray(self.model.vertex_colors)
        self.create_buffers()

    def load_per_vertex_uvs(self) -> Union[np.ndarray, None]:
        """
        Load per-vertex UV coordinates from a PLY file.
        This method reads UV coordinates (s, t) from the vertex data of a PLY file
        specified by `self.filename`. It uses the PlyData library to parse the PLY file
        and extract the UV attributes. If the UV attributes are found, they are combined
        into an Nx2 NumPy array and returned. If the UV attributes are not found or an
        error occurs during loading, the method returns None.
        Returns:
            np.ndarray or None: An Nx2 array of UV coordinates if successful, otherwise None.
        Raises:
            Exception: If an error occurs while reading the PLY file.
        """

        # Use PlyData to load UVs from PLY file
        try:
            ply_data = PlyData.read(self.filename)
            vertex_data = ply_data["vertex"]
            if "s" in vertex_data and "t" in vertex_data:
                s = vertex_data["s"]
                t = vertex_data["t"]
                return np.column_stack((s, t))  # Combine into Nx2 array
            else:
                return None
        except Exception as e:
            print(f"Error loading UVs: {e}")
            return None

    def create_buffers(self) -> None:
        """
        Creates and initializes various buffers for the model.
        This method sets up the following buffers:
        - pos_buffer: Stores the vertex positions.
        - pos_normal_buffer: Stores interleaved vertex positions and normals.
        - pose_normal_uvs_buffer: Stores interleaved vertex positions, normals, and UVs if available.
          If UVs are not available, it falls back to storing only positions and normals.
        Returns:
            None
        """

        self.pos_buffer = self.vertices
        # Flattened positions and normals interleaved
        self.pos_normal_buffer = np.hstack(
            [
                self.model_points,  # positions: [x, y, z]
                self.normals,  # normals: [nx, ny, nz]
            ]
        ).flatten()

        # Add UVs if available
        if self.uvs is not None:
            self.pose_normal_texture_buffer = np.hstack(
                [
                    self.model_points,  # positions: [x, y, z]
                    self.normals,  # normals: [nx, ny, nz]
                    self.uvs,  # UVs: [u, v]
                ]
            ).flatten()
        elif self.colors is not None:
            self.pose_normal_texture_buffer = np.hstack(
                [
                    self.model_points,  # positions: [x, y, z]
                    self.normals,  # normals: [nx, ny, nz]
                    self.colors,  # colors: [r, g, b]
                ]
            ).flatten()
        else:
            self.pose_normal_texture_buffer = (
                self.pos_normal_buffer
            )  # Fallback to position + normals

    def getModelStats(self) -> dict:
        """
        Computes and returns statistical information about the model points.
        This method calculates the mean, maximum, and minimum values for the x, y,
        and z coordinates of the model points. Additionally, it computes the
        bounding box center for the model.
        Returns:
            dict: A dictionary containing the following keys:
                - 'mean_x': Mean value of the x coordinates.
                - 'max_x': Maximum value of the x coordinates.
                - 'min_x': Minimum value of the x coordinates.
                - 'mean_y': Mean value of the y coordinates.
                - 'max_y': Maximum value of the y coordinates.
                - 'min_y': Minimum value of the y coordinates.
                - 'mean_z': Mean value of the z coordinates.
                - 'max_z': Maximum value of the z coordinates.
                - 'min_z': Minimum value of the z coordinates.
                - 'bbox_center': A dictionary with the center of the bounding box:
                    - 'x': Center x coordinate of the bounding box.
                    - 'y': Center y coordinate of the bounding box.
                    - 'z': Center z coordinate of the bounding box.
        """

        mean_x, max_x, min_x = (
            self.model_points[:, 0].mean(),
            self.model_points[:, 0].max(),
            self.model_points[:, 0].min(),
        )
        mean_y, max_y, min_y = (
            self.model_points[:, 1].mean(),
            self.model_points[:, 1].max(),
            self.model_points[:, 1].min(),
        )
        mean_z, max_z, min_z = (
            self.model_points[:, 2].mean(),
            self.model_points[:, 2].max(),
            self.model_points[:, 2].min(),
        )

        out = {
            "mean_x": mean_x,
            "max_x": max_x,
            "min_x": min_x,
            "mean_y": mean_y,
            "max_y": max_y,
            "min_y": min_y,
            "mean_z": mean_z,
            "max_z": max_z,
            "min_z": min_z,
            "bbox_center": {
                "x": max_x - abs(min_x),
                "y": max_y - abs(min_y),
                "z": max_z - abs(min_z),
            },
        }
        return out
