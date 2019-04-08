import numpy as np

Weather = [{"Sunny"}, {"Windy"}, {"Rainy"}]
Parents = [{"Yes"}, {"No"}]
Money = [{"Rich"}, {"Poor"}]
Class = [{"Cinema"}, {"Tennis"}, {"Stay_in"}, {"Shopping"}]

rec = [
    {"Sunny", "Yes", "Rich", "Cinema"}, {"Sunny", "No", "Rich", "Tennis"}, {"Windy", "Yes", "Rich", "Cinema"}, {"Rainy", "Yes", "Poor", "Cinema"}, {"Rainy", "No", "Rich", "Stay_in"}, {
        "Rainy", "Yes", "Poor", "Cinema"}, {"Windy", "No", "Poor", "Cinema"}, {"Windy", "No", "Rich", "Shopping"}, {"Windy", "Yes", "Rich", "Cinema"}, {"Sunny", "No", "Rich", "Tennis"}
]


att = {"Weather": Weather, "Parents": Parents, "Money": Money}


def gini(a):
    s = [len([i for i in rec if a.union(j) <= i]) for j in Class]
    si = np.sum(s)
    s = (s / np.sum(s)) ** 2
    g = 1 - np.sum(s)
    # print(g, a.union(Class[1]))
    return g, si


def gini2(a):
    s = [len([i for i in rec if j <= i]) for j in att[a]]
    si = np.sum(s)
    s = (s / np.sum(s)) ** 2
    g = 1 - np.sum(s)
    # print(g, a.union(Class[1]))
    return g, si


def Gain(a, s):
    G, s = gini2(set(a))
    gi = [gini(i)[0] for i in att[a]]
    si = [gini(i)[1] for i in att[a]]
   # print(gi, si, s)
    gi = [gi[i] * si[i] / s for i in range(len(si))]
    print(np.sum(gi), G)
    gain = G - np.sum(gi)
    print(gain)


def H():
    return p/(p + n) * np.log2((n + p) / p) + n/(p + n) * np.log2((n + p) / n)


def E(a, j):
    n = len(att[a])
    ai = [np.zeros(n) for i in range(len(Decision))]
    for i in range(len(rec)):
        r = rec[i]
        att_ = att[r[j]]
        ai[att_[r[-1]]] += 1


gini({"Sunny"})
Gain("Weather", 10)
