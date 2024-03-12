import numpy as np
import cv2
import random 
import time
# Function to simulate drone movement and rotation
def rotated_to_original(movement_vector_rotated, rotation_angle):
    # Convert rotation angle to radians
    angle_rad = np.radians(rotation_angle)

    # Create a rotation matrix for the inverse rotation
    rotation_matrix_inverse = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)],
                                        [np.sin(-angle_rad), np.cos(-angle_rad)]])

    # Apply the inverse rotation matrix to the rotated movement vector
    movement_vector_original = np.dot(rotation_matrix_inverse, movement_vector_rotated)

    return movement_vector_original

def simulate_drone_movement(image, drone_position, movement_vector, rotation_angle):
    movement_vector = rotated_to_original(movement_vector,-rotation_angle)
    rows, cols, _ = image.shape

    # Create a translation matrix for simulating movement
    translation_matrix = np.float32([[1, 0, movement_vector[0]], [0, 1, movement_vector[1]]])
    
    # Apply translation to the entire image
    moved_image = cv2.warpAffine(image, translation_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Update the drone's position
    drone_position = (
        drone_position[0] + movement_vector[0],
        drone_position[1] + movement_vector[1],
        drone_position[2],
        drone_position[3]
    )

    # Ensure the updated drone position is within the image boundaries
    drone_position = (
        max(0, min(drone_position[0], cols - drone_position[2])),
        max(0, min(drone_position[1], rows - drone_position[3])),
        drone_position[2],
        drone_position[3]
    )
    drone_position = tuple(np.array(drone_position,int))
    # Extract the region of interest (ROI) from the moved image
    roi_image = moved_image[drone_position[1]:drone_position[1] + drone_position[3], drone_position[0]:drone_position[0] + drone_position[2]]

    # Calculate the center of the cropped region
    center_x = drone_position[2] // 2
    center_y = drone_position[3] // 2
    # Create a rotation matrix for simulating rotation around the center of the cropped region
    rotation_matrix = cv2.getRotationMatrix2D((int(center_x), int(center_y)), rotation_angle, 1.0)
    
    # Apply rotation to the ROI
    rotated_roi = cv2.warpAffine(roi_image, rotation_matrix, (drone_position[2], drone_position[3]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Display the updated image with the rotated ROI
    

    return rotated_roi,drone_position

class drone_sim():
    def __init__(self,initial_pos,image,start_time):
        self.time = start_time
        self.drone_pos = initial_pos
        self.drone_map = image
        self.total_rot = 0
    def step(self,movement_vector,rotation_angle,noise = [0.95,1.05]):
        dt = time.time() - self.time
        self.time = time.time()
        movement_vector = tuple(np.array(movement_vector)*dt)
        print(movement_vector,dt)
        rotation_angle = rotation_angle*dt

        movement_vector = np.array(movement_vector)
        movement_vector = tuple([np.random.normal(*noise)*movement_vector[0],np.random.normal(*noise)*movement_vector[1]])
        rotation_angle = np.random.normal(*noise)*rotation_angle

        self.total_rot += rotation_angle

        rotated_roi,self.drone_pos = simulate_drone_movement(self.drone_map, self.drone_pos, movement_vector, self.total_rot)
        return rotated_roi,self.drone_pos