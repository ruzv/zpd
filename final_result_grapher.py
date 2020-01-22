import matplotlib.pyplot as plt
import numpy as np
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
        for i in range(len(lr_r)):
            lr_r[i] = str(lr_r[i])
        net_sizes = [10, 20, 40, 80, 160]
        for i in range(len(net_sizes)):
            net_sizes[i] = str(net_sizes[i])
        for x in range(5):
            performance = []
            for y in range(5):
                performance.append(graph_data[y][x][-1])
            y_pos = np.arange(len(lr_r))
            plt.bar(y_pos, performance)
            plt.xticks(y_pos, lr_r)
            plt.ylabel("end accuriciy")
            plt.title("net:"+actf1+","+actf2+" lr:"+str(net_sizes[x])+" costf:MSEL d:"+data_set)
            plt.savefig("../end_rez_over_lr_plots/net:"+actf1+","+actf2+" lr:"+str(net_sizes[x])+" costf:MSEL d:"+data_set+".png")
            plt.clf()
