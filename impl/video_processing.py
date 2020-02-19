import cv2
import matplotlib.pyplot as plt
import numpy as np

from impl import image_processing as ip

left_edge_predicted = (22, 62)  # (x1, y1)
right_edge_predicted = (309, 41)  # (x2, y2)

SHOW_VIDEO = True


def find_carpet_line(lines_in_image):

    coord_divergence = [0, 0, 0, 0]  # (x1, y1, x2, y2)
    diff_sum = 1000000
    carpet_line = None

    for line_coord in lines_in_image:
        coord_divergence[0] = abs(line_coord[0][0] - left_edge_predicted[0])
        coord_divergence[1] = abs(line_coord[0][1] - left_edge_predicted[1])
        coord_divergence[2] = abs(line_coord[0][2] - right_edge_predicted[0])
        coord_divergence[3] = abs(line_coord[0][3] - right_edge_predicted[1])

        #  finding the line closes to my predicted coordinates
        if sum(coord_divergence) < diff_sum:
            diff_sum = sum(coord_divergence)
            carpet_line = line_coord

    return carpet_line


def detect_line(img):

    # detecting line coordinates using Hough transformation
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    plt.imshow(edges_img, "gray")

    min_line_length = 200

    # Hough transformation
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)

    carpet_line = find_carpet_line(lines)

    if carpet_line is None:
        raise Exception("Carpet line not found")

    x1 = carpet_line[0][0]
    y1 = carpet_line[0][1]
    x2 = carpet_line[0][2]
    y2 = carpet_line[0][3]

    return x1, y1, x2, y2


def detect_cross(x, y, k, n, tolerance_level, upper_bound):

    if detect_cross_with_tolerance(x-tolerance_level, y-tolerance_level, k, n, upper_bound) or \
            detect_cross_with_tolerance(x+tolerance_level, y+tolerance_level, k, n, upper_bound) or \
            detect_cross_with_tolerance(x, y, k, n, upper_bound):

        return True

    return False


def detect_cross_with_tolerance(x, y, k, n, upper_bound):

    yy = k * x + n
    return 0 <= (yy - y) < upper_bound


def locate_people_on_carpet(frame, people_on_carpet):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_bin = ip.adaptive_threshold_gaus(frame_gray, 11, 3)

    img = frame.copy()
    rectangles = ip.get_bounding_rects(img, frame_bin, 10, 10, 80, 80)

    for rectangle in rectangles:
        x, y, w, h = rectangle

        #  if the person is within the carpet coordinates
        if 92 < x < 240 and 70 < y < 400:
            people_on_carpet[(x, y)] = (w, h)

    return people_on_carpet


def process_video(video_path):

    sum_of_people = 0
    k = 0
    n = 0

    # loading video
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    previous_frame = None
    people_on_carpet = {}

    upper_bound = 0
    tolerance_level = 0

    print(video_path)
    while True:  # analyzing video frame per frame
        frame_num += 1
        ret_val, frame = cap.read()

        # if the frame hasn't been caught
        if not ret_val:
            people_on_carpet = locate_people_on_carpet(previous_frame, people_on_carpet)
            break

        # cropping frame
        frame = frame[50:450, 160:500]
        previous_frame = frame

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bin = ip.adaptive_threshold_gaus(frame_gray, 11, 3)

        if frame_num == 1:

            #  locating people who did not cross the line
            people_on_carpet = locate_people_on_carpet(previous_frame, people_on_carpet)

            line_coords = detect_line(frame)
            line_left_edge = line_coords[0]
            line_right_edge = line_coords[2]

            # y = kx + n
            k = (float(line_coords[3]) - float(line_coords[1])) / (float(line_coords[2]) - float(line_coords[0]))
            n = k * (float(-line_coords[0])) + float(line_coords[1])

            print("Detected line:")
            print("k = ", k)
            print("n = ", n)

            if ip.estimate_noise(frame_gray) > 1.825:  # if the image is noisy
                #  reduce tolerance level for line cross
                upper_bound = 1.03
                tolerance_level = 3
            else:
                upper_bound = 1.09
                tolerance_level = 5

        img = frame.copy()
        rectangles = ip.get_bounding_rects(img, frame_bin, 5, 5, 120, 120)

        if SHOW_VIDEO:
            cv2.imshow('frame', img)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

        for rectangle in rectangles:
            x, y, w, h = rectangle

            if (line_left_edge <= x <= line_right_edge) and detect_cross(x, y, k, n, tolerance_level, upper_bound):
                print('Found person in frame: ', frame_num)
                sum_of_people += 1

    print('People found: ', sum_of_people)
    print('People who did not cross the line: ', len(people_on_carpet))
    print()
    cap.release()

    return sum_of_people + len(people_on_carpet)

