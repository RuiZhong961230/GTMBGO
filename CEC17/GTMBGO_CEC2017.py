
# import packages
import os
import numpy as np
from copy import deepcopy
from cec17_functions import cec17_test_func


PopSize = 100  # the number of Pop
DimSize = 10  # the number of variables
LB = [-100] * DimSize  # the maximum value of the variable range
UB = [100] * DimSize  # the minimum value of the variable range
TrialRuns = 30  # the number of independent runs
MaxFEs = 1000 * DimSize  # the maximum number of fitness evaluations

Pop = np.zeros((PopSize, DimSize))  # the coordinates of the individual (candidate solutions)
FitPop = np.zeros(PopSize)  # the fitness value of all Pop
curFEs = 0  # the current number of fitness evaluations
FuncNum = 1  # the serial number of benchmark function
curIter = 0  # the current number of generations
MaxIter = int(MaxFEs / PopSize)
curBest = np.zeros(DimSize)  # the best individual in the current generation
FitBest = 0  # the fitness of the best individual in the current generation
curWorst = np.zeros(DimSize)  # the worst individual in the current generation


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


# initialize the M randomly
def Initialization():
    global Pop, FitPop, curBest, FitBest
    # randomly generate Pop
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitPop[i] = fitness(Pop[i])
    bestIdx = np.argmin(FitPop)
    curBest = Pop[bestIdx].copy()
    FitBest = FitPop[bestIdx]


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def GTMBGO():
    global Pop, FitPop, curBest, FitBest, curWorst
    # update the current worst Pop
    worstIdx = np.argmax(FitPop)
    curWorst = Pop[worstIdx].copy()
    # record the generated Off individual
    Off = np.zeros(DimSize)
    bestIdx = np.argmin(FitPop)
    # battle Phase
    for i in range(PopSize):
        if i == bestIdx:
            target = []
            for j in range(DimSize):
                if np.random.rand() < np.random.normal(0.01, 0.01):
                    target.append(j)
            F = np.random.normal(0.5, 0.1)
            candi = list(range(0, PopSize))
            r1, r2 = np.random.choice(candi, 2, replace=False)

            for j in range(DimSize):
                if np.random.rand() < 0.01:
                    r = np.random.randint(0, PopSize)
                    Off[j] = Pop[bestIdx][j] + F * (Pop[r1][j] - Pop[r][j])
                else:
                    Off[j] = Pop[bestIdx][j] + F * (Pop[r1][j] - Pop[r2][j])
            for j in range(DimSize):
                if j in target:
                    Off[j] = Pop[bestIdx][j]
                else:
                    pass
            Off = Check(Off)
            FitOff = fitness(Off)
            if FitOff < FitPop[i]:
                Pop[i] = Off.copy()
                FitPop[i] = FitOff
        else:
            selectedIdx = np.random.randint(0, PopSize)
            while selectedIdx == i:
                selectedIdx = np.random.randint(0, PopSize)
            if FitPop[i] > FitPop[selectedIdx]:
                space = Pop[selectedIdx] - Pop[i]
                for j in range(DimSize):
                    if np.random.uniform() < 0.5:
                        Off[j] = Pop[i][j] + space[j] * 0.5 * np.random.rand()
                    else:
                        Off[j] = Pop[selectedIdx][j] + space[j] * 0.5 * np.random.rand()
            else:
                space = Pop[i] - Pop[selectedIdx]
                Off = Pop[i] + space * np.cos(2 * np.pi * np.random.rand())
            Off = Check(Off)
            FitOff = fitness(Off)
            if FitOff < FitPop[i]:
                Pop[i] = Off.copy()
                FitPop[i] = FitOff

    bestIdx = np.argmin(FitPop)
    curBest = deepcopy(Pop[bestIdx])
    FitBest = FitPop[bestIdx]


def RunGTMBGO():
    global curFEs, curIter, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curFEs = 0
        curIter = 0
        Initialization()
        BestList.append(FitBest)
        np.random.seed(2024 + 88 * i)
        while curIter <= MaxIter:
            GTMBGO()
            curIter += 1
            BestList.append(FitBest)
        All_Trial_Best.append(BestList)
    np.savetxt("./GTMBGO_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize)
    LB = [-100] * dim
    UB = [100] * dim

    for i in range(1, 31):
        if i == 2:
            continue
        FuncNum = i
        RunGTMBGO()


if __name__ == "__main__":
    if os.path.exists('./GTMBGO_Data/CEC2017') == False:
        os.makedirs('./GTMBGO_Data/CEC2017')
    Dims = [30, 50]
    for Dim in Dims:
        main(Dim)


