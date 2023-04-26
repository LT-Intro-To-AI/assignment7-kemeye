from typing import Tuple
from neural import *


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    # wordRef = {"v-high","high","med","low","more","small","med","big","low","high"}
    # numRef =  {3,2,1,0,6,0,1,2,0,2}
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
        "5more": 6
    }
    tokens = line.split(",")
    del tokens[len(tokens)-1]
    print(tokens)
    # output = [0 if out == 1 else 0.5 if out == 2 else 1]
    con_int = []
    for x in tokens:
        if type(x) == str:
            con_int.append(numWordRef[x])
    out = con_int[0]
    # print(output)
    output = [
        0,0 if out == 0 
        else 0,1 if out == 1 
        else 1,0 if out == 2 
        else 1,1]
    if out == 0:
        output = [0,0]

    inpt = [float(x) for x in con_int[1:]]
    # print(inpt)
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

td = normalize(training_data)

for line in td:
    print(line)

nn = NeuralNet(5, 3, 1)
nn.train(td, iters=1000, print_interval=100, learning_rate=0.1)

for i in nn.test_with_expected(td):
    print(f"desired: {i[1]}, actual: {i[2]}")