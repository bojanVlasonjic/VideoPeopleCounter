from impl import video_processing as vp


# Problematicni - video 4, video 2, video 5


def load_true_values(file_path):
    true_values = {}

    file = open(file_path, "r")
    lines = file.readlines()

    for i in range(1, len(lines)):
        values = lines[i].split(',')
        true_values[values[0]] = int(values[1])

    return true_values


def calc_min_abs_error(true_values, predicted_values):

    if len(true_values) != len(predicted_values):
        raise Exception("True and predicted values must have same length")

    sum = 0
    for i in range(0, len(true_values)):
        sum += abs(predicted_values[i] - true_values[i])

    return sum/len(true_values)


def memorize_mae(file_path, mae):
    file = open(file_path, "w")
    file.write(str(mae))


def get_previous_mae(file_path):
    file = open(file_path, "r")
    return float(file.read())


if __name__ == '__main__':

    people_per_video = load_true_values('data/res.txt')
    predicted_people = []

    for video_name in people_per_video.keys():
        predicted_people.append(vp.process_video('data/' + video_name))

    print('Result: ', predicted_people)
    mae = calc_min_abs_error(list(people_per_video.values()), predicted_people)

    print('Min absolute error: ', mae)
    print('Success: ', mae < 4.6)

    print("Improvement: ", mae < get_previous_mae('data/myres.txt'))
    memorize_mae('data/myres.txt', mae)
