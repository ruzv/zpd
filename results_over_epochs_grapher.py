import matplotlib.pyplot as plt
import numpy as mp
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500

SIGMOID = "SIG"
SOFTMAX = "SOFT"
RELU = "RELU"

data_set = "digits"

for actf1 in [RELU, SOFTMAX, SIGMOID]:
    for actf2 in [RELU, SOFTMAX, SIGMOID]:
        graph_data = [[], [], [], [], []]
        for net_size in [10, 20, 40, 80, 160]:
            file = open("./"+actf1+"_"+actf2+"/net:"+actf1+","+actf2+" ml:"+str(net_size)+" costf:MSEL d:"+data_set, "r")
            k = -1
            for line in file:
                if(line[0:3] == "net"):
                    func = []
                    i = 0
                    k += 1
                if(i > 0 and i < 42):
                    func.append(line)
                if(i == 41):
                    graph_data[k].append(func)
                i += 1
        for x in range(5):
            for y in range(len(graph_data[x])):
                for z in range(len(graph_data[x][y])):
                    p = graph_data[x][y][z]
                    p = list(p)
                    del p[-1]
                    p = "".join(p)
                    p = p.split(" ")
                    p = float(p[1])
                    graph_data[x][y][z] = p
        lr_r = [.125, .5,  1,   2 ,   8,]
        net_sizes = [10, 20, 40, 80, 160]
        for x in range(5):
            plt.title("net:"+actf1+","+actf2+" lr:"+str(lr_r[x])+" costf:MSEL d:"+data_set)
            plt.xlabel("epochs")
            plt.ylabel("networks accuricity")
            for y in range(5):
                plt.plot(range(0, 41), graph_data[x][y], label="middle layer size "+str(net_sizes[y]))
            plt.legend()
            plt.savefig("../new_plots/net:"+actf1+","+actf2+" lr:"+str(lr_r[x])+" costf:MSEL d:"+data_set+".png")
            plt.clf()
