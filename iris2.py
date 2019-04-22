
from sklearn.datasets import load_iris
import numpy as np
import random


def separateByClass(dataset, target):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (target[i] not in separated):
            separated[target[i]] = []
        separated[target[i]].append(vector)
    return separated


def summarize(dataset):
    summaries = [(np.mean(attribute), np.std(attribute))
                 for attribute in zip(*dataset)]
    # ls = [np.cov(attribute, attribute) for attribute in zip(*dataset)]
    # print(ls)
    return summaries


def summarizeByClass(dataset, target):
    separated = separateByClass(dataset, target)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = np.exp(-(np.power(x-mean, 2)/(2*np.power(stdev, 2))))
    return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent


def Priors(target):
    unique_elements, counts_elements = np.unique(target, return_counts=True)
    return counts_elements / np.sum(counts_elements)


def calculateClassProbabilities(summaries, inputVector, priors):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = priors[classValue]
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector, priors):
    probabilities = calculateClassProbabilities(summaries, inputVector, priors)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet, priors):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i], priors)
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1
        # else:
        #     print('Classified as  ', iris.target_names[predictions[i]])
        #     print('Expexted ', iris.target_names[testSet[i]])

    return (correct/float(len(testSet))) * 100.0


def k_set(k):
    round = {}
    size = 150 // k
    full_data = iris.data
    full_target = iris.target
    data_range = np.arange(150)
    for i in range(k):
        random.shuffle(data_range)
        test_idx = data_range[:size]
        train_target = np.delete(iris.target, test_idx)
        train_data = np.delete(iris.data, test_idx, axis=0)
        test_target = iris.target[test_idx]
        test_data = iris.data[test_idx]
        round[i] = [train_data, train_target, test_data, test_target]
        data_range = data_range[size:]
    return round


# prepare model
iris = load_iris()
accuracy = []

for _ in range(30):
    round = k_set(5)
    round_ac = 0

    for i in range(5):
        train_data = round[i][0]
        train_target = round[i][1]
        test_data = round[i][2]
        test_target = round[i][3]
        # print(test_target)
        summaries = summarizeByClass(train_data, train_target)
        class_priors = Priors(train_target)
    # Test
        predictions = getPredictions(summaries, test_data, class_priors)
        round_ac += getAccuracy(test_target, predictions)
    print(round_ac / 5)
    accuracy.append(round_ac / 5)

# Accuracy model


# print(accuracy)
print('Accuracy mean: ', np.mean(accuracy))
print('Accuracy variance: ', np.var(accuracy))
print('Accuracy min: ', np.min(accuracy))
print('Accuracy max: ', np.max(accuracy))
