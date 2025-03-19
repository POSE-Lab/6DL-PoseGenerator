import numpy as np
import cv2
from plyfile import PlyData

def read_models_vertices(model_file: str) -> np.ndarray:
    
    ply_data = PlyData.read(model_file)

    # Access the vertex data
    vertices = ply_data['vertex']  # Structured array containing vertex attributes

    # Extract x, y, z coordinates
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']


    vertex_coords = np.vstack((x, y, z)).T  # Shape: (num_vertices, 3)
    
    return vertex_coords

def build_cuboid_from_model(model: np.ndarray, axis_up: str = 'z') -> np.ndarray:

    min_x = np.min(model[:, 0])
    max_x = np.max(model[:, 0])
    
    min_y = np.min(model[:, 1])
    max_y = np.max(model[:, 1])
    
    min_z = np.min(model[:, 2])
    max_z = np.max(model[:, 2])
    
    cuboid = np.array([
        [min_x, min_y, min_z], # 1
        [min_x, min_y, max_z], # 2
        [max_x, min_y, max_z], # 3
        [max_x, min_y, min_z], # 4
        [min_x, max_y, min_z], # 5
        [min_x, max_y, max_z], # 6
        [max_x, max_y, max_z], # 7
        [max_x, max_y, min_z]  # 8
    ])
    
    if axis_up == 'z':
        return cuboid
    elif axis_up == 'y':
        cuboid[:, [1, 2]] = cuboid[:, [2, 1]]
        print("here")
        return cuboid

def draw_cuboid3D_on_image(image: np.ndarray, cuboid: np.ndarray, pose: np.ndarray, Kmat: np.ndarray, thickness: float, color: tuple) -> np.ndarray:
    
    if image is None:
        raise ValueError("Input image is None.")

    # Project 3D cuboid points to 2D image plane
    projected_points, _ = cv2.projectPoints(cuboid, pose[:3, :3], pose[:3, 3], Kmat, None)
    projected_points = projected_points.reshape(-1, 2).astype(int)

    # Draw lines between the projected points to form the cuboid
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    for edge in edges:
        pt1 = tuple(projected_points[edge[0]])
        pt2 = tuple(projected_points[edge[1]])
        cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    return image


    # Save or display the image
    #output_filename = 'output_' + image_filename
    #cv2.imwrite(output_filename, image)

    #cv2.imshow('Cuboid', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, required=True)
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--pose', type=str, required=True)
    parser.add_argument('--Kmat', type=str, required=True)
    args = parser.parse_args()

    model = read_models_vertices(args.model_file)
    pose = np.load(args.pose)
    Kmat = np.load(args.Kmat)

    cuboid = build_cuboid_from_model(model, axis_up='z')
    draw_cuboid3D_on_image(args.image_file, cuboid, pose, Kmat, thickness=1, color=(255, 0, 0))
