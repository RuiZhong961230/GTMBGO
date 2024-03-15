import numpy as np
from Methods import ERDGk
from os import path
from cec2013lsgo.cec2013 import Benchmark
from CC import CC_exe
from GTMBGO import GTMBGO_exe
import os
import warnings

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    if os.path.exists('./GTMBGO-ERDGk') == False:
        os.makedirs('./GTMBGO-ERDGk')
    Dims = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 905, 905, 1000]

    bench = Benchmark()
    this_path = path.realpath(__file__)

    NIND = 100
    trial = 25
    ERDG_cost = [2998, 2998, 3996, 5330, 5405, 5905, 5512, 8453, 8816, 8794, 9212, 29100, 7583, 8418, 3996]

    for func_num in range(1, 16):
        Dim = Dims[func_num-1]
        FEs = 3000000
        func = bench.get_function(func_num)
        info = bench.get_info(func_num)
        scale = [info["lower"], info["upper"]]

        ERDG_obj_path = path.dirname(this_path) + "/DECC-ERDG/" + str(func_num) + ".csv"
        ERDG_MaxIter = int((FEs - ERDG_cost[func_num-1]) / Dim / NIND)
        ERDG_groups = ERDGk(path.dirname(path.dirname(this_path)) + "/CEC13/ERDGk/f" + str(func_num) + ".mat")

        All_trial = []
        for i in range(trial):
            ERDG_obj = CC_exe(Dim, scale, NIND, ERDG_groups, ERDG_MaxIter, func, GTMBGO_exe)
            All_trial.append([ERDG_obj])
        np.savetxt("./GTMBGO-ERDGk/F" + str(func_num) + ".csv", All_trial, delimiter=",")











