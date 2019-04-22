import numpy as np
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections
import matplotlib.pyplot as plt
import PIL.Image


class Node:
    def __init__(self, name, no_children = 0):
        self.name = name
        self.edges = [""] * no_children
        self.children = [None] * no_children

    def add_node(self, node, edge = 0, index = 0):
        self.edges[index] = edge
        self.children[index] = node


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
                
def dot_node(root, index):
    from_index = index[0]
    dot_data = str(from_index) + " [label=\"" + root.name + "\", fillcolor=\"#ffa500\"];\n"
    c = root.children
    for i in range(len(c)):
            index[0]+= 1
            toindex = index[0]
            if(len(c[i].edges) != 0):
                    dot_data += dot_node(c[i], index)
            else:
                    dot_data = dot_data + str(index[0]) + " [label=\"" + c[i].name + "\", fillcolor=\"#40e0d0\"];\n"
            dot_data = dot_data + str(from_index) + " -> " + str(toindex) + "[labeldistance=2,fontsize= 10, labelangle=45, headlabel=" +root.edges[i]+ "] ;\n"
    return dot_data
                
def Plot_Tree(root):
    dot_data = '''digraph Tree {
            node [shape=ellipse, style="filled, rounded", color="black", fontname=helvetica] ;
            edge [fontname=helvetica] ;\n'''
    dot_data += dot_node(root, [0])
    dot_data += "}"
    graph = pydotplus.graph_from_dot_data(dot_data)
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
    graph.write_png('tree.png')
    img = PIL.Image.open(
    r'tree.png')
    img = np.asarray(img)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()



def Classification(rec, root):
        c = root
        while(len(c.edges) != 0):
                i = attribute.index(c.name)
                j = att[c.name].index(rec[i])
                c = c.children[j]
        return c.name        
display_tree(root)
Plot_Tree(root)
print("Classification of the record ['Sunny', 'Yes', 'Rich'] is ")
print(Classification(['Sunny', 'Yes', 'Rich'], root))
