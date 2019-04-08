import numpy as np


class Node:
    def __init__(self, name, no_children = 0):
        self.name = name
        self.edges = [""] * no_children
        self.children = [None] * no_children

    def add_node(self, node, edge = 0, index = 0):
        self.edges[index] = edge
        self.children[index] = node


# print some values in the tree
# print(root.data[0])
# print(root.child[0].data[0])


Weather = ["Sunny", "Windy", "Rainy"]
Parents = ["Yes", "No"]
Money = ["Rich", "Poor"]
Class = ['Cinema', 'Tennis', 'Stay_in', 'Shopping']
attribute = ["Weather", "Parents", "Money"]

rec = [
    ['Sunny', 'Yes', 'Rich', 'Cinema'], ['Sunny', 'No', 'Rich', 'Tennis'], ['Windy', 'Yes', 'Rich', 'Cinema'], ['Rainy', 'Yes', 'Poor', 'Cinema'], ['Rainy', 'No', 'Rich', 'Stay_in'], [
        'Rainy', 'Yes', 'Poor', 'Cinema'], ['Windy', 'No', 'Poor', 'Cinema'], ['Windy', 'No', 'Rich', 'Shopping'], ['Windy', 'Yes', 'Rich', 'Cinema'], ['Sunny', 'No', 'Rich', 'Tennis']
]

att = {"Weather": Weather, "Parents": Parents, "Money": Money}

flag = [r[-1] for r in rec]

def gini(a, rec):
    m = [np.zeros(len(Class)) for i in att[a]]
    reci = [[] for i in att[a]]
    for r in rec:
        row = att[a].index(r[attribute.index(a)])
        col = Class.index(r[-1])
        reci[row].append(r)
        m[row][col] += 1
    si = np.sum(m, axis=1)
    gi = [1 - np.sum(m[i]**2 / si[i]**2) for i in range(len(si))]
    s = np.sum(m, axis=0)
    if(np.sum(s) == 0):
        return 0, []
    g = 1 - sum([s[i]**2 / sum(s)**2 for i in range(len(s))])
   # print(gi, reci)
    R_Gain = np.sum(gi * si)/np.sum(m)
    return R_Gain, reci

   # id = [[i for i in range(len(rec)) if j in rec[i]] for j in att[a]]

#     s = [len([i for i in rec if a.union(j) <= i]) for j in Class]
#     si = np.sum(s)
#     s = (s / np.sum(s)) ** 2
#     g = 1 - np.sum(s)
#     # print(g, a.union(Class[1]))
#     return g, si


def gini2(a, re):
    m = [np.zeros(len(Class)) for i in att[a]]
    reci = [[] for i in att[a]]
    for r in re:
        row = att[a].index(rec[r][attribute.index(a)])
        col = Class.index(rec[r][-1])
        reci[row].append(r)
        m[row][col] += 1
    si = np.sum(m, axis=1)
    gi = [(1 - np.sum(m[i]**2 / si[i]**2)) if si[i] != 0 else 0  for i in range(len(si))]
    s = np.sum(m, axis=0)
    if(np.sum(s,axis=0) == 0):
        return 0, []
    g = 1 - sum([s[i]**2 / sum(s)**2 for i in range(len(s))])
   # print(gi, reci)
    R_Gain = np.sum(gi * si)/np.sum(m)
    return R_Gain, reci


def gini3(a, re):
    m = []
    reci = [[] for i in att[a]]
    #for r in re:
    unique_elements, counts_elements = np.unique([flag[i] for i in re ], return_counts=True)    
    si = np.sum(m, axis=1)
    gi = [1 - np.sum(m[i]**2 / si[i]**2) for i in range(len(si))]
    s = np.sum(m, axis=0)
    if(np.sum(s) == 0):
        return 0, []
    g = 1 - sum([s[i]**2 / sum(s)**2 for i in range(len(s))])
   # print(gi, reci)
    R_Gain = np.sum(gi * si)/np.sum(m)
    return R_Gain, reci


# min = 100
# node = ()
# for i in attribute:
#     g, r = gini(i, rec)
#     if g < min:
#         min = g
#         node = (i, r)


def id3(att_list, rec_list):
        if(len(att_list) == 0):
               return Node(name = rec[rec_list[0]][-1]) 
        min = 100
        node = ()
        for i in att_list:
          g, r = gini2(attribute[i], rec_list)   
          print(attribute[i],r)
          if g < min:
                min = g
                node = (i, r)        
        a = attribute[node[0]]
        #print(a)
        # if(a == "Weather"):
        #         print(node[1][0])
        root = Node(name = a, no_children = len(att[a]))
        att_list.remove(node[0])
        for i in range(len(att[a])):
                u = set()
                for j in node[1][i]:
                        u.add(rec[j][-1])
                if(len(u) == 1) :
                        root.add_node(Node(name = u.pop()),att[a][i], i)         
                        continue
                root.add_node(id3(att_list, node[1][i]),att[a][i], i)
        return root

a_list = [0,1,2]
r = [0,1,2,3,4,5,6,7,8,9]
root = id3(a_list, r)
# print(root.name)
# for i in root.children:
#         print(i.name)
#         if(len(i.edges) != 0):
#                 print(i.edges[0])

def display_tree(root):
        print(root.name)
        c = root.children
        for i in range(len(c)):
                print(root.edges[i])
                if(len(c[i].edges) != 0):
                      # print(c[i].edges[0])
                        display_tree(c[i])
                else:
                        print(c[i].name)
                
               

display_tree(root)

#root = id3(att, rec)
# g, r = gini2("Weather", [1, 2, 4, 5])
# print(g)
# Gain("Weather", 10)
