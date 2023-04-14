from neural import NeuralNet

# print("\n\nTraining SQ\n\n")
# sq_training_data = [
#     ([0.2], [0.04]),
#     ([0.3], [0.09]),
#     ([0.5], [0.25]),
#     ([0.7], [0.49]),
#     ([0.1], [0.01]),
#     ([0.9], [0.81]),
# ]
# sqn = NeuralNet(1, 20, 1)
# sqn.train(sq_training_data)

# print()
# print(sqn.test_with_expected(sq_training_data))
# print(sqn.evaluate([0.66]))
# print(sqn.evaluate([0.95]))

x_or_trainingdata = [
    ([0,0], [0]),
    ([1,0],[1]),
    ([0,1],[1]),
    ([1,1],[0]),
]

xorn = NeuralNet(2,20,1)

xorn.train(x_or_trainingdata)

print()
print(xorn.test_with_expected(x_or_trainingdata))