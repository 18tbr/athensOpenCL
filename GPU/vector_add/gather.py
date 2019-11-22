# Gathering
from subprocess import run, check_output

for n in [4000,40000,100000,200000,300000,400000,600000]:
    for mode in ["copy","map"]:
        for worksize in [1,2,4]:
            path = "outfile/outfile_{}_{}_{}.txt".format(n,mode,worksize)
            command = "./vector_add {} {} {} > {}".format(n,mode,worksize,path)
            print(command)
            run(command,shell=True)
