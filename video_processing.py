import numpy as np
import cv2
import Projekat3.image_processing as ip
import matplotlib.pyplot as plt

# (180, 113) - (470, 92)
# (22, 62) - (309, 41)
left_edge_predicted = (22, 62)  # (x1, y1)
right_edge_predicted = (309, 41)  # (x2, y2)


def find_carpet_line(lines_in_image):

    coord_divergence = [0, 0, 0, 0]  # (x1, y1, x2, y2)
    diff_sum = 1000000
    carpet_line = None

    for line_coord in lines_in_image:
        coord_divergence[0] = abs(line_coord[0][0] - left_edge_predicted[0])
        coord_divergence[1] = abs(line_coord[0][1] - left_edge_predicted[1])
        coord_divergence[2] = abs(line_coord[0][2] - right_edge_predicted[0])
        coord_divergence[3] = abs(line_coord[0][3] - right_edge_predicted[1])

        if sum(coord_divergence) < diff_sum:
            diff_sum = sum(coord_divergence)
            carpet_line = line_coord

    return carpet_line


def detect_line(img):

    # detekcija koordinata linije koristeci Hough transformaciju
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    plt.imshow(edges_img, "gray")

    # minimalna duzina linije
    min_line_length = 200

    # Hough transformacija
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


def detect_cross(x, y, k, n):

    tolerance_level = 5  # line thickness in px

    if detect_cross_with_tolerance(x-tolerance_level, y-tolerance_level, k, n) or \
            detect_cross_with_tolerance(x+tolerance_level, y+tolerance_level, k, n) or \
            detect_cross_with_tolerance(x, y, k, n):

        return True

    return False


def detect_cross_with_tolerance(x, y, k, n):

    yy = k * x + n
    return 0 <= (yy - y) < 1.12


def process_video(video_path):
    # procesiranje jednog videa
    # priprema pomocnih promenljivih
    sum_of_people = 0
    k = 0
    n = 0

    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)  # indeksiranje frejmova

    print(video_path)
    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        ret_val, frame = cap.read()

        # ako frejm nije zahvacen
        if not ret_val:
            break

        # isecanje frejma
        frame = frame[50:450, 160:500]

        if frame_num == 1:  # ako je prvi frejm, detektuj liniju
            line_coords = detect_line(frame)
            line_left_edge = line_coords[0]
            line_right_edge = line_coords[2]

            # odredjivanje parametara jednacine prave y = kx + n
            k = (float(line_coords[3]) - float(line_coords[1])) / (float(line_coords[2]) - float(line_coords[0]))
            n = k * (float(-line_coords[0])) + float(line_coords[1])
            print("Detected line:")
            print("k = ", k)
            print("n = ", n)

            # coloring line
            # cv2.line(frame, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), (255, 0, 0), 2)
            # ip.display_image(frame, '', 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # plt.imshow(frame_gray, "gray")
        frame_bin = ip.adaptive_threshold_gaus(frame_gray, 11, 3)
        # plt.imshow(frame_bin, "gray")

        img = frame.copy()
        rectangles = ip.get_bounding_rects(img, frame_bin, 5, 5, 80, 80)
        #plt.imshow(img)

        for rectangle in rectangles:
            x, y, w, h = rectangle

            if (line_left_edge <= x <= line_right_edge) and detect_cross(x, y, k, n):
                #ip.display_image(frame_bin, video_path, 2)
                #ip.display_image(img, video_path, 2)
                sum_of_people += 1

    print('People found: ', sum_of_people)
    print()
    cap.release()
    return sum_of_people
