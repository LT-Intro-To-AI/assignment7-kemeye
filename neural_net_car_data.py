from typing import Tuple
from neural import *


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    numWordRef = {
        "vhigh": 3,
        "high": 2,
        "med": 1,
        "low": 0,
        "more": 6,
        "small": 0,
        "big": 2,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        "5more": 5
    }
    tokens = line.split(",")
    del tokens[len(tokens)-1]
    # print(tokens)
    con_int = []
    for x in tokens:
        con_int.append(int(numWordRef[x]))
    out = int(con_int[0])
    # print(out)
    output = [0.0]
    if out == 0:
        output = [0.0]
    elif out == 1:
        output = [0.33]
    elif out == 2:
        output = [0.66]
    elif out == 3:
        output = [1.0]

    # output = [0.0, 0.0]
    # if out == 0:
    #     output = [0.0, 0.0]
    # elif out == 1:
    #     output = [0.0, 1.0]
    # elif out == 2:
    #     output = [1.0, 0.0]
    # elif out == 3:
    #     output = [1.0, 1.0]

    # print(f"output: {output}")

    inpt = [float(x) for x in con_int[1:]]
    # print(f"input: {inpt}")
    # print(f"output tuple {(inpt, output)}")
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("car.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines()[::10] if len(line) > 4]

# f = open("car.txt", "r")
# fList = f.readlines()
# for x in range(len(training_data)):
#     print(training_data[x])
#     print(fList[x])
td = normalize(training_data)

# for line in td:
#     print(line)

nn = NeuralNet(5, 10, 1)
nn.train(td, iters=200, print_interval=10, learning_rate=0.9)

for i in nn.test_with_expected(td):
    print(f"predicted: {i[2]}, actual: {i[1]}")
