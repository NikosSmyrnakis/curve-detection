import cv2
import numpy as np
import math
import time
from simulation import simulate_drone_movement
from image_analysis import image_analysis
from simulation import drone_sim
from drone_control import get_movment
from drone_tools import zoom

import warnings

# Suppress RankWarning
warnings.filterwarnings("ignore", category=np.RankWarning)

def text_to_image(image,text,where):
    cv2.putText(image, text, (result_image.shape[1] - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, where), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


mask_attempts = 0

if __name__ == "__main__":
    # Load an example image
    image = cv2.imread("maps/curve4.png")
    image = np.clip(image + np.random.normal(0, 0.8, image.shape).astype(np.uint8), 0, 255)
    img2 = image.copy()
    movement_vector = (0,0); rotation_speed = 0; start_time = time.time(); fps = 10000; i=0; paxos = None

    #setup environment
    env = drone_sim((0, 270, 400, 400),image)
    old_curve = None
    while True:

        #apply the movement and go one step in the simulation 
        frame,drone_position =  env.step(movement_vector, rotation_speed)#,[0.8,1.2])
        #image analysis
        edges,result_image, gray,paxos,slope_error,y_error,test_curve = image_analysis(frame,paxos,old_curve)
        if test_curve is not None or mask_attempts==2:
            mask_attempts = 0
            old_curve = test_curve
        else:
            mask_attempts+=1
        #get the movment
        rotation_speed, movement_vector = get_movment(slope_error,y_error)


        ###Imshow the resutls else
        #continue
        i+=1
        if i%30==0 and i!=0:
            i = 0; elapsed_time = time.time() - start_time
            start_time = time.time()
            fps = 1 / (elapsed_time / 30)
        
        text_to_image(result_image,f"FPS: {fps:.2f}",30)
        cv2.imshow("Line Following", np.vstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),result_image,cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]))
        pos = np.array(drone_position[0:2],int) + np.array(drone_position[2:4],int)//2
        cv2.drawMarker(img2, tuple(pos), (0, 255, 0), cv2.MARKER_CROSS, 15, 100)
        
        if not i%10:
            cv2.imshow("Drone View", img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the OpenCV window
    cv2.destroyAllWindows()





