import numpy as np
import pandas as pd
import time, sys
from matplotlib import pyplot as plt

class StatTestsPlot:
    def general_plotter(self, x, y, xlabel, ylabel, txt = None, leg = None):
        for a in y:
            plt.plot(x, a)
        if leg is not None:
            plt.legend(leg, loc='upper right')
        if txt is not None:
            pass
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    #Copied from https://stackoverflow.com/questions/45394981/how-to-generate-list-of-unique-random-floats-in-python
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    def gen_uniq_floats(self, lo, hi, n):
        out = np.empty(n)
        needed = n
        while needed != 0:
            arr = np.random.uniform(lo, hi, needed)
            uniqs = np.setdiff1d(np.unique(arr), out[:n-needed])
            out[n-needed: n-needed+uniqs.size] = uniqs
            needed -= uniqs.size
        np.random.shuffle(out)
        return out.tolist()
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

class Node:  
    def __init__(self, kv:(float,int) = (None, None), height:int = 0):
        self.kv = kv
        self.next = [None]*(1+height)                                             #Next layer nodes

class SkipList:
    def __init__(self, height:int = 0):
        self.head = Node((None, None), height)
        self.layer = height  
        
    def multi_random_layers(self, n:int, p:float) -> [int]:
        for x in range(n):
            layer = 0
            while np.random.choice(2, 1, p=[1-p, p]):
                layer += 1
            yield (layer)

    ''' As the name already suggests, add_multi_nodes adds multiple nodes to the skip-list, where kv
        is an 2D-Matrix, in which every entry is an tuple, consisting of (key, value)

        For every entry in said matrix, we first calculate the new height, which will
        be HIGHT if specified and if not, we call multi_random_layers to calculate
        every max layer in respect to p for each entry in our Matrix.

        We then calculate the list of Nodes we have to travel to get to our destination, 
        where we can insert our key (see search() for more information). We have reached the end of 
        our last layer if hlayers is None/our referenced key isn't already in use, we therefore
        can insert our key there.

        [NOTE/BUG]: Commented lines don't work, they should dynamically change the ceiling :/
                    Could be bc of the messed up ref list

        We create a new Node with the entry in kv and the respective layer we calculated earlier.
        Last but not least, we have to rearrange our node-references to fit our new node in'''

    def add_multi_nodes(self, kv:[float,int], HIGHT:int = 0, p:float = 0.5) -> None:                                 #Adds [a,b,c,...] nodes (multiple nodes to reduce overhead, due to pythons function calling)
        global compE, compADD, compDEL, compSER
        MAX_LAYER = [HIGHT]*len(kv) if HIGHT != [0] else list(self.multi_random_layers(len(kv),p))
        compE, compADD = compE +1, compADD +1
        c = 0
        for x in kv:
            ref, hlayers = self.search(x, 1)
            if hlayers is None or self.head.kv != x[0]:
                compE, compADD = compE +1.5, compADD +1.5                               #rough average comparisons due to lazy evaluation
#                if MAX_LAYER[c] > self.layer: 
#                    for n in range(self.layer+1, MAX_LAYER[c]+1):
#                        ref[n] = self.head
#                    self.layer = MAX_LAYER[c] 
                node = Node(x, MAX_LAYER[c]) 
                for n in range(MAX_LAYER[c]+1): 
                    node.next[n] = ref[n].next[n] 
                    ref[n].next[n] = node
            c += 1
            
    ''' Works almost the same as multi_nodes; We get an array of keys, which we give
        search() to get a list where the specified node entries are saved.
        With this list, we can check if the to-be targeted note exists and if the
        answer was positive, we rearrange the references like in multi-nodes and remove
        every entry part by part.
        
        [NOTE]  Commented out code removes unnecessary None-Lines. Has to be commented out
                until we fixed the ceiling bug'''
    

    def multi_delete(self, key:[int]) -> None: 
        global compE, compADD, compDEL, compSER
        for x in key:
            ref, hlayers = self.search(x,2)
            if hlayers is not None and hlayers.kv[0] == x: 
                compE, compDEL = compE+2, compDEL+2                                 #Could also be 3 comparisons, depending on how lazy evaluation works with and
                for n in range(self.layer+1): 
                    if ref[n].next[n] != hlayers: 
                        compE, compDEL = compE+1, compDEL+1
                        break
                    ref[n].next[n] = hlayers.next[n] 
#                while(self.layer>0 and self.head.next[self.layer] == None): 
#                    self.layer -= 1
    

    ''' The search-function gets an array, consisting of tuples where the keys/values are stored respectively.
        The Head is the current pointer and the overwriter is needed to save the found nodes.
        Due to the fact, that certain functions (e.g. delete) call search with only an array of keys,
        we have to normalize the shape of kv with an dummy variable (-1 in our case).

        From here on happens nothing special; We start at the top layer and go down if 
        head.next[n] is None or if the key of head.next[n] is >= than the referenced key.
        Overwriter will now save the state where our head stopped last (e.g. None or the key we searched for)

        If search was called from another function, then it returns overwriter + head.next[0]
        to use them for further operations and if search was called directly, then it will output
        the respective tuple/nearest tuple concerning the key (always rounded up) '''

    def search(self, kv:[float,int] or [float], REFERENCE_FLAG:int = 0) -> [[Node],Node] or (String, Node):
        global compE, compADD, compDEL, compSER
        head = self.head
        overwriter = [None]*(1+self.layer)                                          #Init layer-reference array
        kv = kv if REFERENCE_FLAG == 1 else [kv,-1]
        compE, compSER = compE+1, compSER+1  
        for n in range(self.layer, -1, -1):                                         #Top-Layer -> Lowest Layer
            while head.next[n] is not None and head.next[n].kv[0] < kv[0]:
                compE, compSER = compE+2, compSER+2 
                head = head.next[n]                                                 #True -> head will move to next node in same layer
            overwriter[n] = head                                                    #False -> move to next layer
        if (REFERENCE_FLAG):
            compE, compSER = compE+1, compSER+1
            return (np.array([overwriter, head.next[0]], dtype="object"))
        return("Searched->", kv if head.kv is not None and head.kv == kv else head.next[n].kv if head.next[n] is not None else ([-99,-99]))
        
    ''' Creates and prints the dataframe of our referenced object
    '''
    def table(self) -> None: 
        data = []
        header = self.head
        for layer in range(self.layer+1): 
            kvs = []
            node = header.next[layer] 
            while(node != None): 
                kvs.append(["K:"+str(node.kv[0])+" V:"+str(node.kv[1])])
                node = node.next[layer] 
            if data != []:
                kvs += [None]*((len(data[0])-len(kvs)))
            data.append(kvs)
        col = list(map(str,list(range(len(data[0])))))
        idx = list(map(str,list(range(self.layer+1))))
        print(pd.DataFrame(np.array(data, dtype="object"),
                columns=["Node: "+ x for x in col], index = ["Layer: "+ x for x in idx]))

'''                 ---DRIVER CODE---
    TEST_FLAG <- [0|1]  : 0 = predefined testsuite, 1 = testsuite with a lot of parameters to play with
    hight <- 1,..,n     : Specifies max height of skip-list 
    F_HIGHT <- 1,..,n   : Forces how high the added nodes have to be placed
    d_size <- 1,..,n    : How many (key,value) tuples should be generated for testing in testsuite 1
    i <- 1,..,n         : How many times testsuite 1 should be run in respect to p (e.g. i = 10, len(p) = 3 => 30 iterations)
    p <- [0<=float<=1]  : Probability if an copy of an tuple should be created in a higher layered skiplist
    VERB_FLAG <- [0|1]  : Prints out useful graphs/informations concerning runtime/comparisons etc.
'''
def main(TEST_FLAG:int = 0, hight:int = 10, F_HIGHT:int = 2, d_size:int = 10000, i:int = 5, p:[float] = [0.25, 0.5, 0.75], VERB_FLAG:int = 1) -> None:
    global compE, compADD, compDEL, compSER
    compE, compADD, compDEL, compSER = 0, 0, 0, 0 
    
    #Basic testsuite to test every function manually
    if (TEST_FLAG == 0):                                                                    
        lst = SkipList(hight) 
        lst.add_multi_nodes([[0.5,2],[5,20],[4,12],[17,81]], [0], p[1]) 
        lst.add_multi_nodes([[20,2]], F_HIGHT)
        lst.table()
        print("------------------------------------------------------------------------")
        print(lst.search(0.55))         #Not existent -> Next should [4, 12]
        print(lst.search(0.5))          #existent, should print [0,5, 2]
        print("------------------------------------------------------------------------")
        lst.multi_delete([0.5,5,17]) 
        lst.table()
        print("------------------------------------------------------------------------") 
        lst.add_multi_nodes([[0.5,2],[0.6,17], [1,3]])
        lst.table() 
        print("------------------------------------------------------------------------")
        VERB_FLAG and print("Comparisons: ", compE)
        
    #Advanced testsuite with d_size samples, i*len(p) repititions, repeated deletes/searches and graphs on how well everything performed
    if (TEST_FLAG):                                                                         
        stp = StatTestsPlot()   
        t = list(zip(stp.gen_uniq_floats(0, 2*10**5, d_size), np.random.default_rng().choice(d_size*5, size=d_size, replace=False)))
        t = [x.tolist() for x in np.array_split(t, int(d_size/1000))]
        sub_counted_c, overall_time = [], []
        for x in p:
            for it in range(i):
                subtimer_n = time.time()
                lst = SkipList(100)
                for z in t:
                    foundT = []
                    randomT = stp.gen_uniq_floats(0, 2*10**5, int(d_size/100))
                    lst.add_multi_nodes(z, [0], x)
                    for rT in randomT:
                        foundT.append(lst.search(rT)[1][0])
                    lst.multi_delete(foundT[:500])
                sub_counted_c.append([compE, compADD, compDEL, compSER])
                compE, compADD, compDEL, compSER = 0, 0, 0, 0
                overall_time.append(time.time()-subtimer_n)
        if (VERB_FLAG):
            pyc, pycm, pyt, pytm = [], [], [], []
            for x in p:
                pyc.append([x[0] for x in sub_counted_c][:i])
                pycm.append(np.mean([x[0] for x in sub_counted_c][:i]))
                pyt.append([x for x in overall_time][:i])
                pytm.append(np.mean([x for x in overall_time][:i]))

                sub_counted_c = sub_counted_c[i:]
                overall_time = overall_time[i:]

            stp.general_plotter(list(range(i)), pyc, "Iteration", "Comparisons", None, ["p = "+x for x in list(map(str, p))])
            stp.general_plotter(list(range(i)), pyt, "Iteration", "Runtime [s]", None, ["p = "+x for x in list(map(str, p))])
            print(pd.DataFrame(data = {"Mean Runtime": pytm, "Mean Comparisons": pycm}, index = ["[p] = "+ x for x in list(map(str, p))]))

print("-----Testsuite 1-----\n\n\r")
main()
print("\n\n\r-----Testsuite 2-----\n\n\r")
main(1)