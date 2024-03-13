import cv2
import numpy as np
import math
import time

def curve_masking(points, curve_coefficients,image_shape,enabled = True):
    if enabled:
        points_x = list(map(lambda x:x[0],points))
        points_y = list(map(lambda x:x[1],points))
        if curve_coefficients is not None:
            diffs = np.abs(np.polyval(curve_coefficients, points_x) - points_y)
            threshold = np.percentile(diffs, 80)
            return np.where(diffs<threshold)[0]
            #return np.where(diffs<image_shape[0]/5)[0]
        else:
            return list(range(len(points)))
    else:
        return list(range(len(points)))



def cv2_curve_points(points,degree=4):
    #(points)
    x = list(map(lambda x:x[0],points))
    y = list(map(lambda x:x[1],points))
    coefficients = np.polyfit(x, y, degree)

    # Generate x values for the curve
    x_curve = np.linspace(min(x), max(x), 100)

    # Calculate y values using the polynomial coefficients
    y_curve = np.polyval(coefficients, x_curve)
    return x_curve,y_curve,coefficients

def find_curve(points, num_iterations=100, threshold=20,degree=2):
    best_model = None
    best_inliers = None
    best_res = None
    points = np.array(points)
    if len(points)<5:
        return None,None,None
    for _ in range(num_iterations):
        # Randomly select five points to form a model
        indices = np.random.choice(len(points), (2*len(points))//3, replace=False)
        sample_points = points[indices]

        # Fit a 4th-degree polynomial model to the random sample
        model = np.polyfit(sample_points[:, 0], sample_points[:, 1], degree)

        # Calculate the residuals from all points to the model
        residuals = np.abs(np.polyval(model, points[:, 0]) - points[:, 1])
        threshold = np.percentile(residuals, 90)
        #print(residuals)
        # Count inliers based on the threshold
        inliers = np.where(residuals <= threshold)[0]

        # Update the best model if the current model has more inliers
        if best_res is None or best_res > np.sum(residuals[inliers]):
            best_model = model
            best_inliers = inliers
            best_res = np.sum(residuals[inliers])

    # Fit the final model using all inliers
    #final_model = np.polyfit(points[best_inliers, 0], points[best_inliers, 1], degree)
    points = points[best_inliers]

    #return list(map(lambda x:x[0],points)),list(map(lambda x:x[1],points))

    if len(points)>0:
        #print(len(best_inliers))
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



def flip_lines(edges,lines):
    height, width = edges.shape[:2]
    center_x = width // 2
    center_y = height // 2
    #print('--->',lines.shape)
    rotated_lines = np.array([[[y1,x1,y2,x2]] for [[x1, y1, x2, y2]] in lines], dtype=np.int32)
    return rotated_lines


def image_analysis(image,paxos = None,old_curve = None):
    #image[image<80] = image[image<80]*0.5
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = gray*2
    #gray = adjust_brightness(gray)
    #gray = sharpen_image_gray(gray)
    # Apply Canny edge detection
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    #edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    #edges = cv2.Laplacian(gray, ksize=3, ddepth=cv2.CV_16S)
    #edges = cv2.convertScaleAbs(edges)
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    #edges = cv2.GaussianBlur(edges, (15,15), 0)
    step = 10
    angle_limit = 4
    min_dis = 10
    max_dis = 90
    if paxos!=None:
        #print(paxos)
        min_dis = 0.6*paxos
        max_dis = 1.4*paxos
    checks = 10
    # Apply HoughLinesP to detect lines
    dlines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=20, maxLineGap=10)
    
    if dlines is None or len(dlines)<5:
        return no_lines_imshow(edges,image,gray)
    
    for g in range(2):
        if g==1:
            width, height = edges.shape[:2]
            lines = flip_lines(edges,dlines)
        else:
            height, width = edges.shape[:2]
            lines = dlines
        #print(len(lines))
        # Draw lines on the original image
        result_image = image.copy()
        
        lines = np.array(list(map(lambda x:list(form_line(x[0])),lines)))

        points = np.array(list(map(lambda x:[(x[0]+x[2])/2,(x[1]+x[3])/2],lines)))
        masked_idxs = curve_masking(points,old_curve,image.shape)
        lines = lines[masked_idxs]
        if lines is None:
            return no_lines_imshow(edges,image,gray)
        #lines = lines[np.argsort(lines[:,0])]



        points = [[] for i in range(width//step)]

        #width//10


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
                    lines1.append(item[0])
                    lines2.append(item[1])
                else:
                    lines2.append(item[0])
                    lines1.append(item[1])

        #lines = lines[list(set([item for sublist in points for item in sublist]))]

        if g==0:
            lines1_g0 = lines1
            lines2_g0 = lines2
            lines_g0 = lines
        if g==1:
            lines1 = flip_lines(edges,np.array(lines1).reshape(-1,1,4))
            lines1 = lines1_g0+[np.array(l[0]) for l in lines1]
            lines2 = flip_lines(edges,np.array(lines2).reshape(-1,1,4))
            lines2 = lines2_g0+[np.array(l[0]) for l in lines2]
            lines = np.concatenate((lines_g0,lines))
        if lines1 is not None and len(lines1)>1:
            #print(lines1)
            for line in lines1:
                #print(line)
                x1, y1, x2, y2 = line
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        if lines2 is not None and len(lines2)>1:
            for line in lines2:
                x1, y1, x2, y2 = line
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        
        points1 = np.array(list(map(lambda x:[(x[0]+x[2])/2,(x[1]+x[3])/2],lines1)))
        points2 = np.array(list(map(lambda x:[(x[0]+x[2])/2,(x[1]+x[3])/2],lines2)))
        



        
        if len(points1)>4:
            if g==0:
                continue
            return get_imshows(result_image,image,gray,edges,lines,points1,points2)
        elif g==1:
            print("here",len(points1))
            return no_lines_imshow(edges,image,gray)
            

def get_imshows(result_image,image,gray,edges,lines,points1,points2):
    #print(points2)
    points_m = np.int32((np.array(points1)+np.array(points2))/2)
    paxos = np.mean(np.abs(points1[:,1]-points2[:,1]))
    x_curve, y_curve, poly = find_curve(points_m,degree = 3)
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
        return edges,result_image,gray,paxos,slope_error,y_error,poly

def no_lines_imshow(edges,image,gray):
    return edges,image,gray,None,None,None,None