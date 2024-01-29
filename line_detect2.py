import cv2
import numpy as np
import math
import time

def cv2_curve_points(points,degree=3):
    #(points)
    x = list(map(lambda x:x[0],points))
    y = list(map(lambda x:x[1],points))
    coefficients = np.polyfit(x, y, degree)

    # Generate x values for the curve
    x_curve = np.linspace(min(x), max(x), 100)

    # Calculate y values using the polynomial coefficients
    y_curve = np.polyval(coefficients, x_curve)
    return x_curve,y_curve,coefficients

def find_curve(points, num_iterations=100, threshold=30,degree=4):
    best_model = None
    best_inliers = None
    points = np.array(points)
    for _ in range(num_iterations):
        # Randomly select five points to form a model
        indices = np.random.choice(len(points), len(points)//3, replace=False)
        sample_points = points[indices]

        # Fit a 4th-degree polynomial model to the random sample
        model = np.polyfit(sample_points[:, 0], sample_points[:, 1], degree)

        # Calculate the residuals from all points to the model
        residuals = np.abs(np.polyval(model, points[:, 0]) - points[:, 1])

        # Count inliers based on the threshold
        inliers = np.where(residuals < threshold)[0]

        # Update the best model if the current model has more inliers
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers

    # Fit the final model using all inliers
    #final_model = np.polyfit(points[best_inliers, 0], points[best_inliers, 1], degree)
    points = points[best_inliers]
    #return list(map(lambda x:x[0],points)),list(map(lambda x:x[1],points))

    if len(points)>0:
        return cv2_curve_points(points,degree)
    else:
        return None,None,None
def slope(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1
#
    if delta_x == 0:
        return float('inf')  # Vertical line, undefined slope
#
    slope_radians = math.atan(delta_y / delta_x)
    slope_degrees = math.degrees(slope_radians)
#
    return slope_degrees

def diss(line1,line2):
    x1, y1, x2, y2 = line1
    m = (y1-y2)/(x1-x2)
    b1= y1 - m*x1
    x1, y1, x2, y2 = line2
    b2= y1 - m*x1
    return abs(b2-b1)/np.sqrt(m*m+1)


def draw_points(image, points, color=(0, 255, 0)):
    # Draw markers at specified points on the image
    for point in points:
        cv2.circle(image, (point), radius=5, color=(255 , 255), thickness=5)

def form_line(line):
    x1, y1, x2, y2 = line
    if x1>x2:
        return [x2,y2,x1,y1]
    else:
        return line

def sharpen_image_gray(image, sharpen_strength=3):
    # Define a sharpening kernel with adjustable weights
    kernel = np.array([[-1, -1, -1],
                       [-1,  8 + sharpen_strength, -1],
                       [-1, -1, -1]])/2

    # Apply the kernel convolution to the grayscale image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image


def image_analysis(image,paxos = None):
    #image[image<80] = image[image<80]*0.5
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = gray*2
    #gray = adjust_brightness(gray)
    #gray = sharpen_image_gray(gray)
    # Apply Canny edge detection
    #gray = cv2.GaussianBlur(gray, (9, 9), 2)
    #edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    #edges = cv2.Laplacian(gray, ksize=3, ddepth=cv2.CV_16S)
    #edges = cv2.convertScaleAbs(edges)
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    #edges = cv2.GaussianBlur(edges, (15,15), 0)
    step = 10
    angle_limit = 4
    min_dis = 20
    max_dis = 80
    if paxos!=None:
        min_dis = 0.8*paxos
        max_dis = 1.2*paxos
    checks = 10
    # Apply HoughLinesP to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=10)
    
    if lines is None:
        return edges,image,gray,None
    #print(len(lines))
    # Draw lines on the original image
    result_image = image.copy()
    
    lines = np.array(list(map(lambda x:list(form_line(x[0])),lines)))

    #lines = lines[np.argsort(lines[:,0])]



    points = [[] for i in range(edges.shape[1]//step)]

    #edges.shape[1]//10


    idx = -1
    if lines is not None and len(lines)>1:
        for line in lines:
            idx+=1
            x1, y1, x2, y2 = line
            for i in (np.array(list(range(x1 - (x1-1)%step,x2,step)))-1)//step:
                points[i].append(idx)

    i = -1

    for sample_idxs in points:
        i+=1
        status = 0
        for l in range(checks):
            if len(sample_idxs)>=2:
                sample = lines[sample_idxs]
                slopes = np.array(list(zip(range(len(sample)),[slope(*list(line)) for line in sample])))
                slopes = slopes[np.argsort(slopes[:,1])]
                diff = np.diff(slopes[:,1])
                if not np.isnan(diff).all() and abs(np.nanmin(diff))<angle_limit:
                    min_idx = np.nanargmin(diff)
                    line2idx = [int(slopes[min_idx,0]),int(slopes[min_idx+1,0])]
                    dis = diss(*sample[line2idx])
                    #print(dis)
                    #input()
                    if dis>min_dis and dis<max_dis:
                        points[i] = [sample_idxs[line2idx[0]],sample_idxs[line2idx[1]]]
                        status =1
                        break
                    else:
                        sample_idxs = np.delete(sample_idxs,[line2idx])
                        #points[i] = []
                else:
                    #points[i] = []
                    break
            else:
                points[i] = []
                break
        if status==0:
            points[i] = []
        




    lines1 = []
    lines2 = []
    for idxs in points:
        item = lines[idxs]
        if len(item)>0:
            if (item[0][1]+item[0][3])/2 > (item[1][1]+item[1][3])/2:
                lines1.append(item[0]);lines2.append(item[1])
            else:
                lines2.append(item[0]);lines1.append(item[1])

    #lines = lines[list(set([item for sublist in points for item in sublist]))]

    if lines1 is not None and len(lines1)>1:
        for line in lines1:
            x1, y1, x2, y2 = line
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if lines2 is not None and len(lines2)>1:
        for line in lines2:
            x1, y1, x2, y2 = line
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    
    points1 = np.array(list(map(lambda x:[(x[0]+x[2])/2,(x[1]+x[3])/2],lines1)))
    points2 = np.array(list(map(lambda x:[(x[0]+x[2])/2,(x[1]+x[3])/2],lines2)))
    
    
    points_m = np.int32((np.array(points1)+np.array(points2))/2)
    if len(points_m)>5:
        paxos = np.mean(np.abs(points1[:,1]-points2[:,1]))
        x_curve, y_curve, poly = find_curve(points_m,degree = 4)
        if poly is not None:
            poly_der = np.polyder(poly)
            y_error = result_image.shape[0]/2-np.polyval(poly,result_image.shape[1]/2)
            slope_error = np.degrees(np.polyval(poly_der,result_image.shape[1]/2))
            
            cv2.putText(result_image, f"Y error: {y_error:.2f}", (result_image.shape[1] - cv2.getTextSize(f"Y error: {y_error:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(result_image, f"Slope (mires): {slope_error:.2f}", (result_image.shape[1] - cv2.getTextSize(f"Slope (mires): {slope_error:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(result_image, f"Paxos grammhs: {paxos:.2f}", (result_image.shape[1] - cv2.getTextSize(f"Paxos grammhs: {paxos:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(result_image, f"Lines: {len(lines):.2f}", (result_image.shape[1] - cv2.getTextSize(f"Lines: {len(lines):.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            
            curve_points = np.column_stack((x_curve, y_curve)).astype(int)
            cv2.polylines(result_image, [curve_points], isClosed=False, color=(50, 125, 255), thickness=int(4*paxos/5))
            return edges,result_image,gray,paxos

    return edges,image,gray,None

    #print(points_m)
    #draw_points(result_image,points_m)
    
# Display the result

# Read the image
if False:
    image = cv2.imread("test5.jpg")
    edges,result_image = image_analysis(image)
    cv2.imshow("Line Following", np.vstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),result_image]))
    # Wait for a key event and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)


def zoom(cv_image, scale=15):
    height, width, _ = cv_image.shape
    # print(width, 'x', height)
    # prepare the crop
    centerX, centerY = int(height / 2), int(width / 2)
    radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)

    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY

    cv_image = cv_image[minX:maxX, minY:maxY]
    cv_image = cv2.resize(cv_image, (width, height))

    return cv_image


start_time = time.time()
fps = 10000
i=0
paxos = None
while True:
    i+=1
    if i%30==0 and i!=0:
        i = i%10000000
        elapsed_time = time.time() - start_time
        start_time = time.time()
        fps = 1 / (elapsed_time / 30)

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = zoom(frame,15)
    if not ret:
        print("Error: Could not read frame.")
        break
    
    #edges,result_image = frame,frame
    edges,result_image, gray,paxos = image_analysis(frame,paxos)
    cv2.putText(result_image, f"FPS: {fps:.2f}", (result_image.shape[1] - cv2.getTextSize(f"FPS: {fps:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    #cv2.imshow("Line Following", result_image)
    #cv2.imshow("Line Following", np.vstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),result_image]))
    cv2.imshow("Line Following", np.vstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),result_image,cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]))
    # Wait for a key event and close the windows


    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()