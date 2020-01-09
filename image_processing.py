import cv2
import matplotlib.pyplot as plt


def get_bounding_rects(image_orig, image_bin, min_height=15, min_width=20, max_height=100, max_width=100):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        if area > 100 and h < max_height and h > min_height and w > min_width and w < max_width:
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bounding_rects.append([x, y, w, h])

    return bounding_rects


def adaptive_threshold_gaus(image_gs, block_size=3, constant=5, inv=False):
    # params: (src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])
    if inv:
        return cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)
    return cv2.adaptiveThreshold(image_gs, 255,  + cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)


def display_image(image, title="", wait_time=3, color=False):
    """
    :param image: image to display
    :param title: the title that will be displayed
    :param wait_time: how long will the image be showm
    :param color: is the image in color
    :return: void
    """
    plt.title(title)

    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

    plt.pause(wait_time)
