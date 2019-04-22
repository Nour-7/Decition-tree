# import numpy as np
# # n, m = input().split()
# # array = list(map(int, input().split()))
# # a = set(map(int, input().split()))
# # b = set([1, 2, 3])
# # c = str(b)
# a = {}
# a['abc'] = 0
# print(a)
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict

d = defaultdict(list)
n, m = map(int, input().split())
for i in range(1, n + 1):
    d[input()].append(i)
for i in range(m):
    x = input()
    ls = d[x]
    if len(ls) == 0:
        print(-1)
    else:
        for i in ls:
            print(i, end=' ')
        print()
