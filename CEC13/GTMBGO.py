import numpy as np
from copy import deepcopy


def GTMBGO_exe(subPop, group, func, context, MaxIter, scale, Optimum):
    NIND = len(subPop)
    tPop = np.array([context] * NIND)

    # Embedding the sub-population to context vector
    for i in range(len(group)):
        tPop[:, group[i]] = subPop[:, i]
    tPopFit = np.zeros(NIND)
    for i in range(NIND):
        tPopFit[i] = func(tPop[i])
        Optimum = min(tPopFit[i], Optimum)

    # Optimization
    for maxiter in range(MaxIter):
        Off = deepcopy(context)

        bestIdx = np.argmin(tPopFit)
        for i in range(NIND):
            if i == bestIdx:
                target = []
                for j in group:
                    if np.random.rand() < np.random.normal(0.01, 0.01):
                        target.append(j)
                F = np.random.normal(0.5, 0.1)
                candi = list(range(0, len(tPop)))
                r1, r2 = np.random.choice(candi, 2, replace=False)

                for j in group:
                    if np.random.rand() < 0.01:
                        r = np.random.randint(0, len(tPop))
                        Off[j] = tPop[bestIdx][j] + F * (tPop[r1][j] - tPop[r][j])
                    else:
                        Off[j] = tPop[bestIdx][j] + F * (tPop[r1][j] - tPop[r2][j])
                for j in group:
                    if j in target:
                        Off[j] = tPop[bestIdx][j]
                    else:
                        pass
                CheckIndi(Off, scale)
                FitOff = func(Off)
                Optimum = min(FitOff, Optimum)
                if FitOff < tPopFit[i]:
                    tPop[i] = Off.copy()
                    tPopFit[i] = FitOff
            else:
                selectedIdx = np.random.randint(0, len(tPop))
                while selectedIdx == i:
                    selectedIdx = np.random.randint(0, len(tPop))
                if tPopFit[i] > tPopFit[selectedIdx]:
                    space = tPop[selectedIdx] - tPop[i]
                    for j in group:
                        if np.random.uniform() < 0.5:
                            Off[j] = tPop[i][j] + space[j] * 0.5 * np.random.rand()
                        else:
                            Off[j] = tPop[selectedIdx][j] + space[j] * 0.5 * np.random.rand()
                else:
                    space = tPop[i] - tPop[selectedIdx]
                    Off = tPop[i] + space * np.cos(2 * np.pi * np.random.rand())
                CheckIndi(Off, scale)
                FitOff = func(Off)
                Optimum = min(FitOff, Optimum)
                if FitOff < tPopFit[i]:
                    tPop[i] = Off.copy()
                    tPopFit[i] = FitOff

    bestPop = tPop[np.argmin(tPopFit)]
    bestSubPop = np.zeros(len(group))
    returnPop = np.zeros_like(subPop)
    for i in range(len(group)):
        bestSubPop[i] = bestPop[group[i]]
        returnPop[:, i] = tPop[:, group[i]]
    return bestSubPop, returnPop, Optimum


def CheckIndi(Indi, scale):
    range_width = scale[1] - scale[0]
    Dim = len(Indi)
    for i in range(Dim):
        if Indi[i] > scale[1]:
            n = int((Indi[i] - scale[1]) / range_width)
            mirrorRange = (Indi[i] - scale[1]) - (n * range_width)
            Indi[i] = scale[1] - mirrorRange
        elif Indi[i] < scale[0]:
            n = int((scale[0] - Indi[i]) / range_width)
            mirrorRange = (scale[0] - Indi[i]) - (n * range_width)
            Indi[i] = scale[0] + mirrorRange
        else:
            pass


