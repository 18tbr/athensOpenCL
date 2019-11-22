# Gathering
from subprocess import run, check_output

for n in [4,40,100,200,300,400,600]:
    for mode in ["copy","map"]:
        for worksize in [1,2,4]:
            path = "outfile/outfile_{}_{}_{}.txt".format(n,mode,worksize)
            command = "./matrix_multiply {} {} {} > {}".format(n,mode,worksize,path)
            print(command)
            run(command,shell=True)
