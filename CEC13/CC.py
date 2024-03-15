import numpy as np


def CC_exe(Dim, scale, NIND, groups, MaxIter, func, Optimizer):
    Pop, Fit = InitialPop(Dim, scale, NIND, func)
    Optimum = min(Fit)
    context = np.zeros(Dim)
    for t in range(MaxIter):
        for i in range(len(groups)):
            subPop = SubPop(Pop, groups[i])
            bestSubPop, returePop, Optimum = Optimizer(subPop, groups[i], func, context, len(groups[i]), scale, Optimum)
            for j in range(len(groups[i])):
                context[groups[i][j]] = bestSubPop[j]
                Pop[:, groups[i][j]] = returePop[:, j]
    return min(func(context), Optimum)


def InitialPop(Dim, scale, NIND, func):
    Pop = np.zeros((NIND, Dim))
    Fit = np.zeros(NIND)
    for i in range(NIND):
        for j in range(Dim):
            Pop[i][j] = np.random.uniform(scale[0], scale[1])
        Fit[i] = func(Pop[i])
    return Pop, Fit


def SubPop(Pop, group):
    subpop = np.zeros((len(Pop), len(group)))
    for i in range(len(group)):
        subpop[:, i] = Pop[:, group[i]]
    return subpop