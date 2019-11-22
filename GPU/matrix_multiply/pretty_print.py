import matplotlib.pyplot as plt
import os
import numpy as np

# Gathering
# from subprocess import run, check_output

# for n in [4,40,100,200,300,400,600]:
#     for mode in ["copy","map"]:
#         for worksize in [1,2,4]:
#             path = "outfile/outfile_{}_{}_{}.txt".format(n,mode,worksize)
#             command = "./matrix_multiply {} {} {} > {}".format(n,mode,worksize,path)
#             print(command)
#             run(command,shell=True)

LN = [4,40,100,200,300,400,600]
LMAP = [[0 for i in range(7)],[0 for i in range(7)],[0 for i in range(7)]]
LCPU = [0 for i in range(7)]
LCOPY = [[0 for i in range(7)],[0 for i in range(7)],[0 for i in range(7)]]


n_mapping = {4 : 0,40 : 1,100 : 2,200 : 3,300 : 4,400 : 5,600 : 6}

for root, dirs, files in os.walk("outfile", topdown=False):
    for name in files:
        fileSplit = name.split("_")
        n = int(fileSplit[1])
        mode = fileSplit[2]
        worksize = int(fileSplit[3].split(".")[0])
        # Reading timings from file
        lines = []
        with open(os.path.join(root,name),"r") as source:
            lines = source.readlines()
        GPU_time = 0
        for line in lines:
            if line.startswith("Time to get the vectors : "):
                GPU_time += int(line[len("Time to get the vectors : "):])
            elif line.startswith("GPU took : "):
                GPU_time += int(line[len("GPU took : "):])
            elif line.startswith("Reading time : "):
                GPU_time += int(line[len("Reading time : "):])
        if mode == "map":
            if worksize == 1:
                LMAP[0][n_mapping[n]] = GPU_time
            elif worksize == 2:
                LMAP[1][n_mapping[n]] = GPU_time
            elif worksize == 4:
                LMAP[2][n_mapping[n]] = GPU_time
        if mode == "copy":
            if worksize == 1:
                LCOPY[0][n_mapping[n]] = GPU_time
            elif worksize == 2:
                LCOPY[1][n_mapping[n]] = GPU_time
            elif worksize == 4:
                LCOPY[2][n_mapping[n]] = GPU_time
        if mode == "copy" and worksize == 1:  # just reading cpu once
            CPU_time = 0
            for line in lines:
                if line.startswith("CPU took : "):
                    CPU_time += int(line[len("CPU took : "):])
            LCPU[n_mapping[n]] = CPU_time



Llegend = []
LlegendLabel = []

lblcpu, = plt.plot(LN,np.log(LCPU))
Llegend.append(lblcpu)
LlegendLabel.append("CPU")

compt = 0
for Lmap in LMAP:
    lblmap, = plt.plot(LN,np.log(Lmap))
    Llegend.append(lblmap)
    LlegendLabel.append("GPU - map - {}".format(2**compt))
    compt += 1

compt = 0
for Lcopy in LCOPY:
    lblcopy, = plt.plot(LN,np.log(Lcopy))
    Llegend.append(lblcopy)
    LlegendLabel.append("GPU - copy - {}".format(2**compt))
    compt += 1

plt.legend(Llegend,LlegendLabel)
plt.title("Calcul du produit matriciel avec différents paramètres (échelle semi-log)")
plt.show()
