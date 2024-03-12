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