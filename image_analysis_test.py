import cv2
import numpy as np
import math
import time
from simulation import simulate_drone_movement
from image_analysis import image_analysis
from simulation import drone_sim
from drone_control import get_movment
from drone_tools import zoom
import time

import warnings

# Suppress RankWarning
warnings.filterwarnings("ignore", category=np.RankWarning)

image = cv2.imread("vertical.png")
#image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
edges,result_image, gray,paxos,slope_error,y_error,test_curve = image_analysis(image,None,None)
cv2.imshow("Line Following", np.vstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),result_image,cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]))
cv2.waitKey(0)