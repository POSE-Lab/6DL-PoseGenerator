from math import radians
import pyrr
import math as mt
import numpy as np


class camera:
    """
    A class to represent a camera for 3D rendering and manipulation.
    Attributes
    ----------
    phi : float
        The azimuthal angle in spherical coordinates.
    theta : float
        The polar angle in spherical coordinates.
    radius : float
        The distance from the camera to the target point.
    target : np.array
        The target point the camera is looking at.
    cameraPos : np.array
        The position of the camera in 3D space.
    tranformation : pyrr.Matrix44
        The transformation matrix for the camera view.
    Methods
    -------
    setParams(currPhi, currTheta, currDist):
        Sets the parameters for the camera.
    getClipingPlanes(stats):
        Returns the near and far clipping planes based on the model statistics.
    update_camera_parameters():
        Updates the camera parameters based on the current angles and radius.
    transform2Zup():
        Transforms the camera position to align with the Z-up coordinate system.
    """

    def __init__(self, phi=0, theta=0, radius=6000, target=np.array([0, 0, 0])) -> None:
        self.phi = phi
        self.theta = theta
        self.radius = radius
        self.target = target
        self.cameraPos = np.array([1, 2, 5]).astype(np.float64)
        self.tranformation = pyrr.Matrix44.look_at(
            self.cameraPos + np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
        )

    def setParams(self, currPhi, currTheta, currDist):
        """
        Set the parameters for the camera.
        Args:
            currPhi (float): The current azimuthal angle (phi) of the camera.
            currTheta (float): The current polar angle (theta) of the camera.
            currDist (float): The current distance (radius) from the camera to the target.
        Description:
            This method sets the camera's spherical coordinates, which include the azimuthal angle (phi),
            the polar angle (theta), and the distance (radius) from the camera to the target.
        """

        self.phi = currPhi
        self.theta = currTheta
        self.radius = currDist

    def getClipingPlanes(self, stats):
        """
        Calculate the clipping planes for a camera based on provided statistics.
        Args:
            stats (dict): A dictionary containing the statistics of the scene.
                          Expected keys are "max_z" and "min_z".
        Returns:
            tuple: A tuple containing the far and near clipping planes.
                   - far (float): The far clipping plane, calculated as 1000 times the absolute value of "max_z".
                   - near (float): The near clipping plane, calculated as the absolute value of "min_z".
        """

        far = 1000 * abs(stats["max_z"])
        near = abs(stats["min_z"])
        return far, near

    def update_camera_parameters(self):
        """
        Updates the camera parameters based on the current values of theta, phi, and radius.
        This method calculates the camera's position in 3D space using spherical coordinates
        and updates the camera's transformation matrix to look at the origin (0, 0, 0).
        Returns:
            int: Returns -1 if theta is out of the valid range [0, 180), otherwise None.
        """

        if self.theta < 0 or self.theta >= 180:
            return -1
        if self.theta == 0:
            self.cameraPos[0] = (
                self.radius
                * mt.cos(radians(float(self.phi)))
                * mt.sin(radians(float(0.00001)))
            )
            self.cameraPos[1] = (
                self.radius
                * mt.sin(radians(float(self.phi)))
                * mt.sin(radians(float(0.00001)))
            )
            self.cameraPos[2] = self.radius * mt.cos(radians(float(0.00001)))
        else:

            self.cameraPos[0] = (
                self.radius
                * mt.cos(radians(float(self.phi)))
                * mt.sin(radians(float(self.theta)))
            )
            self.cameraPos[1] = (
                self.radius
                * mt.sin(radians(float(self.phi)))
                * mt.sin(radians(float(self.theta)))
            )
            self.cameraPos[2] = self.radius * mt.cos(radians(float(self.theta)))  # z,y

        centerVec = np.array([0, 0, 0])
        LookAt = pyrr.Matrix44.look_at(
            self.cameraPos + centerVec, centerVec, np.array([0, 0, 1])
        )

        self.tranformation = LookAt

    def transform2Zup(self):
        """
        Transforms the camera position to a Z-up coordinate system.
        This method swaps the y and z coordinates of the camera position to
        convert it from a Y-up to a Z-up coordinate system. It then calculates
        a new look-at matrix for the camera using the updated position and sets
        it as the transformation matrix.
        Attributes:
            self.cameraPos (pyrr.Vector3): The current position of the camera.
            self.tranformation (pyrr.Matrix44): The transformation matrix representing
                                                the camera's orientation and position.
        """

        temp = self.cameraPos.y
        self.cameraPos.y = self.cameraPos.z
        self.cameraPos.z = temp

        centerVec = np.array([0, 0, 0])
        LookAt = pyrr.Matrix44.look_at(
            self.cameraPos + centerVec, centerVec, np.array([0, 0, 1])
        )

        self.tranformation = LookAt
