from collections import defaultdict
import sioux_info as I
import sioux_falls_flow as FLOW
import numpy as np
four_node = [#0  1  2  3
            [ 0, 1, 3, 0], # 0
            [ 0, 0, 0, 0], # 1
            [ 0, 2, 0, 4], # 2
            [ 0, 5, 0, 0]  # 3
            ]


sioux_falls = [
#1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
[0,1,2,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
[3,0,0,0,0,4,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
[5,0,0,6,0,0,0,0,0,0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
[0,0,8,0,9,0,0,0,0,0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
[0,0,0,11,0,12,0,0,13,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #5
[0,14,0,0,15,0,0,16,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #6
[0,0,0,0,0,0,0,17,0,0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0], #7
[0,0,0,0,0,19,20,0,21,0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0], #8
[0,0,0,0,23,0,0,24,0,25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #9
[0,0,0,0,0,0,0,0,26,0, 27, 0, 0, 0, 28, 29, 30, 0, 0, 0, 0, 0, 0, 0], #10
[0,0,0,31,0,0,0,0,0,32, 0, 33, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #11
[0,0,35,0,0,0,0,0,0,0, 36, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #12
[0,0,0,0,0,0,0,0,0,0, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39], #13
[0,0,0,0,0,0,0,0,0,0, 40, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 42, 0], #14
[0,0,0,0,0,0,0,0,0,43, 0, 0, 0, 44, 0, 0, 0, 0, 45, 0, 0, 46, 0, 0], #15
[0,0,0,0,0,0,0,47,0,48, 0, 0, 0, 0, 0, 0, 49, 50, 0, 0, 0, 0, 0, 0], #16
[0,0,0,0,0,0,0,0,0,51, 0, 0, 0, 0, 0, 52, 0, 0, 53, 0, 0, 0, 0, 0], #17
[0,0,0,0,0,0,54,0,0,0, 0, 0, 0, 0, 0, 55, 0, 0, 0, 56, 0, 0, 0, 0], #18
[0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 57, 0, 58, 0, 0, 59, 0, 0, 0, 0], #19
[0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 60, 61, 0, 62, 63, 0, 0], #20
[0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 65, 0, 66], #21
[0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 68, 69, 0, 70, 0], #22
[0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 72, 0, 73], #23
[0,0,0,0,0,0,0,0,0,0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 75, 0, 76, 0], #24
]


# in order to use siox_falls, you have to subtract by one. the results
# given you have to add by 1 just because in sioux_falls it starts at 1

class Graph:
    def __init__(self, graph: [[int]]):
        self._graph = graph
        self._nodes = len(graph) # gives how many nodes there are on the graph
        self._paths = []
        self._linkPath = []

    def connected(self, node: int) -> [int]:
        """
        Given a node, finds all the nodes that are connected
        to it in the given graph
        """
        res = []
        for neighbor in range(len(self._graph[node])):
            if self._graph[node][neighbor] >= 1:
                res.append(neighbor)
        return res

    def dfs(self, orig: int, dest:int, visited: [bool],
            path: [int]):
        """
        Helper function to find the paths using recursion for DFS
        """
        visited[orig] = True
        path.append(orig)

        if orig == dest:
            # found the destination, path should hold the correct path
            self._paths.append(path.copy())
        else:
            # need to find a neighbor for recursion
            for neighbor in self.connected(orig):
                if visited[neighbor] == False:
                    self.dfs(neighbor, dest, visited, path)

        path.pop()
        visited[orig] = False
        
    
    
    def allPaths(self, orig: int, dest: int):
        """
        Finds all paths for the graph given an origin and destination
        """
        visited = [False]*self._nodes # mark everything as unvisited
        path = [] # no path yet
        self._paths.clear()
        self.dfs(orig, dest, visited, path)
        return self._paths

    def linkPathMatrix(self):
        """
        *only use after calling allPaths on the graph*
        Converts dfs paths to link-path matrix for use in optimization
        """
        #trying to convert dfs paths to link-path matrix
        linkPath = [[0] * len(self._paths) for i in range(76)] #hard coded # of edges for four_node
        pathnum = 0
        for path in self._paths:
            for edge in range(len(path) - 1): 
                orig = path[edge]
                dest = path[edge + 1]
                edgenum = self._graph[orig][dest] - 1
                linkPath[edgenum][pathnum] = 1
            pathnum += 1
        self._linkPath = linkPath
        return linkPath

    def pathWeights(self, x, linkPath):
        """
        *only use after calling linkPathMatrix on the graph*
        Calculates the ta(xa) wieghts of each path from the link path matrix
        Uses all-or-nothing assignment
        x = flow through the network
        """
        
        #path weights in ta(xa) fashion; only really applies to node_four atm
        edgeWeights = []
        pathWeights = [0] * len(linkPath[0])
        for edge in range(len(linkPath)):
            tf = I.falls_info[edge + 1][0]/I.falls_info[edge + 1][3]
            ca = I.falls_info[edge + 1][1]*I.falls_info[edge + 1][2]
            edgeWeights.append(tf * (1 + 0.15*(pow((x[edge]/(ca)), 4))))
        for path in range(len(linkPath[0])):
            for edge in range(len(linkPath)):
                if(linkPath[edge][path] == 1):
                    pathWeights[path] += edgeWeights[edge]
        return pathWeights

def diff_of_arrays(a1: [int], a2: [int]) -> [int]:
    res = [0] * len(a1)
    for i in range(len(res)):
        res[i] = a1[i] - a2[i]

    return res

def add_of_arrays(a1: [int], a2: [int]) -> [int]:
    res = [0] * len(a1)
    for i in range(len(res)):
        res[i] = a1[i] + a2[i]

    return res

def mult_of_arrays(c: int, a2: [int]) -> [int]:
    res = [0] * len(a2)
    for i in range(len(res)):
        res[i] = c * a2[i]

    return res

def link_performance(x: int, edge: int) -> int:
    '''
    this calculates the link travel time for a given link
    '''
    alpha=0.15
    beta=4

    tf = I.falls_info[edge + 1][0]/I.falls_info[edge + 1][3]
    ca = I.falls_info[edge + 1][1]*I.falls_info[edge + 1][2]
    return tf * (1 + alpha*(pow((x/(ca)), beta)))


def obj_prime(x: [int], y: [int], alpha: int) -> int:
    '''
    The obj_prime is a defined formula used for bisection
    '''
    ret = 0
    t = [0] * len(x)
    for i in range(len(t)):
        t[i] = (y[i]-x[i])*link_performance(x[i]+alpha*(y[i]-x[i]), i)
        ret = ret + t[i]

    #print(f"obj prime is {ret}")
    return ret
    

def bisection(f_prime, x: [int], y: [int], a=0, b=1, sigma=.00001):
    '''
    performe a bisection for line search to ensure that the obj function
    is within the scope
    '''
    if a > b:
        a,b = b,a
        
    fa,fb = f_prime(x,y,a),f_prime(x,y,b)
    if fa==0:
        b = a
    if fb==0:
        a = b

    while b-a > sigma:
        c = (a+b)/2
        f = f_prime(x,y,c)

        # do the similar thing as above
        if f==0:
            a,b = c,c
        elif (f > 0 and fa > 0) or (f < 0 and fa < 0):
            a = c
        else:
            b = c

    return (a,b)

        

"""Start of Frank-Wolfe Algorithm for Sioux_Falls"""
#Initialization Start
n = 0
g = Graph(sioux_falls)
x = [0] * 76
#link path for all 16 combinations (majority of initial runtime)
#Initialization will take some time to calculate, maybe a min or two just for the 12 shortest paths and their respective
#L-P matrixes, thankfully we never have to calculate them after that so all things reguarding them afterwards are
#O(n)
ODPair = [
#Dest: 6     7     18     20    Src:
    [(0,5),(0,6),(0,17),(0,19)], #1
    [(1,5),(1,6),(1,17),(1,19)], #2
    [(2,5),(2,6),(2,17),(2,19)], #3
    [(12,5),(12,6),(12,17),(12,19)]] #13

# Each elemnent will have the Link-Path Matrix for the corrosponding Source and Destination in the diagram above

LPM = [[0] * 4 for i in range(4)]
for src in range(4):
    for dest in range(4):
        O = ODPair[src][dest][0]
        D = ODPair[src][dest][1]
        g.allPaths(O, D)
        LPM[src][dest] = g.linkPathMatrix()

#All-or-nothing initialization
FOD = [[0] * 4 for i in range(4)]
XOD = [[0] * 4 for i in range(4)]
for src in range(4):
    for dest in range(4):
        PW = g.pathWeights(x, LPM[src][dest])
        FOD[src][dest] = [0] * len(PW)
        FOD[src][dest][PW.index(min(PW))] = 2 #assigning 2 units of flow to the shortest path
        XOD[src][dest] = np.dot(FOD[src][dest], np.asarray(LPM[src][dest]).transpose()) #path flow -> link flow

y = [0] * 76
for link in range(76):
    for src in range(4):
        for dest in range(4):
            y[link] += XOD[src][dest][link]
iter_count = 0
x_prior = x
#x = y

a,b = bisection(obj_prime,x,y) 
c = (a+b)/2
x = add_of_arrays(x, mult_of_arrays(c,(diff_of_arrays(y,x))))
#Initialization Done
"""END OF MY PART FOR NOW, NEED LINE SEARCH FOR LOOP TO WORK CORRECTLY, WILL GET EVERYTHING UP TILL THEN READY ANYWAYS"""
while(np.linalg.norm(diff_of_arrays(x, x_prior),2) >= 0.0000001):
    x_prior = x
    #Pretty much just copy pasted from initialization
    FOD = [[0] * 4 for i in range(4)]
    XOD = [[0] * 4 for i in range(4)]
    for src in range(4):
        for dest in range(4):
            PW = g.pathWeights(x, LPM[src][dest])
            FOD[src][dest] = [0] * len(PW)
            OD_PAIR_NOW = ODPair[src][dest]
            FOD[src][dest][PW.index(min(PW))] = FLOW.falls_flow_demand[OD_PAIR_NOW] #assigning 2 units of flow to the shortest path
            XOD[src][dest] = np.dot(FOD[src][dest], np.asarray(LPM[src][dest]).transpose()) #path flow -> link flow
    y = [0] * 76
    for link in range(76):
        for src in range(4):
            for dest in range(4):
                y[link] += XOD[src][dest][link]

    # starting the line search 
    a,b = bisection(obj_prime,x,y) 
    c = (a+b)/2
    #print(f"c is {c}")
    x = add_of_arrays(x, mult_of_arrays(c,(diff_of_arrays(y,x))))
    iter_count += 1
    #print(f"x is {x} after the iteration")
    #print(f"xa is {x_prior} after the iteration")
    #print(f"norm is {np.linalg.norm(diff_of_arrays(x, x_prior),2)}")
    n += 1
    if n > 500:
        break


''' write x into a file '''
write_x = open("x_file.txt", "w")
count = 1
write_x.write("x is [")
for i in x:
    write_x.write(str(i) + ", ")
write_x.write("]")
    
print(x)

''' write the OD pairs into another file '''
OD_Pairs = open("od_pairs.txt", "w")

# OD Pair for 1->6
OD_Pairs.write("Path travel time for OD pair 1->6 is [")
for i in g.pathWeights(x, LPM[0][0]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")


# OD Pair for 1->7
OD_Pairs.write("Path travel time for OD pair 1->7 is [")
for i in g.pathWeights(x, LPM[0][1]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 1->18
OD_Pairs.write("Path travel time for OD pair 1->18 is [")
for i in g.pathWeights(x, LPM[0][2]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 1->20
OD_Pairs.write("Path travel time for OD pair 1->20 is [")
for i in g.pathWeights(x, LPM[0][3]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 2->6
OD_Pairs.write("Path travel time for OD pair 2->6 is [")
for i in g.pathWeights(x, LPM[1][0]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 2->7
OD_Pairs.write("Path travel time for OD pair 2->7 is [")
for i in g.pathWeights(x, LPM[1][1]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 2->18
OD_Pairs.write("Path travel time for OD pair 2->18 is [")
for i in g.pathWeights(x, LPM[1][2]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 2->20
OD_Pairs.write("Path travel time for OD pair 2->20 is [")
for i in g.pathWeights(x, LPM[1][3]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 3->6
OD_Pairs.write("Path travel time for OD pair 3->6 is [")
for i in g.pathWeights(x, LPM[2][0]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 3->7
OD_Pairs.write("Path travel time for OD pair 3->7 is [")
for i in g.pathWeights(x, LPM[2][1]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 3->18
OD_Pairs.write("Path travel time for OD pair 3->18 is [")
for i in g.pathWeights(x, LPM[2][2]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 3->20
OD_Pairs.write("Path travel time for OD pair 3->20 is [")
for i in g.pathWeights(x, LPM[2][3]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 13->6
OD_Pairs.write("Path travel time for OD pair 13->6 is [")
for i in g.pathWeights(x, LPM[3][0]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 13->7
OD_Pairs.write("Path travel time for OD pair 13->7 is [")
for i in g.pathWeights(x, LPM[3][1]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 13->18
OD_Pairs.write("Path travel time for OD pair 13->18 is [")
for i in g.pathWeights(x, LPM[3][2]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")

# OD Pair for 13->20
OD_Pairs.write("Path travel time for OD pair 13->20 is [")
for i in g.pathWeights(x, LPM[3][3]):
    OD_Pairs.write(str(i) + ", ")
OD_Pairs.write("]\n\n")



write_x.close()
OD_Pairs.close()



