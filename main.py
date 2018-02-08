from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import random

#problem Definition]

def f(x):
    #change function by changing f# the number below to (1-9)
    return f1(x)
#High Conditioned Elliptic Function
def f1(x):
    sum = 0.0
    for i in range(1, len(x) + 1):
        sum += (10 ** 6) ** ((i - 1) / (len(x) - 1)) * x[i - 1] ** 2
    return sum

#Bent cigar function
def f2(x):
    sum = 0.0
    sum += x[0] ** 2
    for i in range(2, len(x) + 1):
        sum += x[i - 1] ** 2
    sum *= (10 ** 6)
    return sum

#Discus Function
def f3(x):
    sum = 0.0
    sum += (x[0] ** 2) * (10 ** 6)
    for i in range(2, len(x) + 1):
        sum += x[i - 1] ** 2
    return sum

#Rosenbrok's Saddle
def f4(x):
    sum = 0.0
    for i in range(len(x) - 1):
        sum += 100 * ((x[i] ** 2) - x[i + 1]) ** 2 + (1 - x[i]) ** 2
    return sum

#Ackley's Function
def f5(x):
    sum1, sum2 = 0.0, 0.0
    for i in range(0, len(x)):
        sum1 += x[i] ** 2
    sum1 = sum1 / float(len(x))
    for i in range(0, len(x)):
        sum2 += np.cos(2 * np.pi * x[i])
    sum2 = sum2 / float(len(x))

    # Calculate first exp
    exp1 = -20.0 * (np.e ** (-0.2 * sum1))
    exp2 = np.e ** sum2

    # Calculate final result
    result = exp1 - exp2 + 20 + np.e
    return result


def f6(x):
    sum1, sum2, sum3 = 0.0, 0.0, 0.0
    a = 0.5
    b = 3
    kmax = 20
    for i in range(len(x)):
        for k in range(0, kmax):
            sum2 = sum2 + (a ** k) * np.cos(2 * np.pi * (b ** k) * (x[i] + 0.5))
            sum3 = sum2 + (a ** k) * np.cos(2 * np.pi * (b ** k) * 0.5)
    sum1 = sum1 + sum2 - (len(x) * sum3)
    return sum1


#Greivank's Function
def f7(x):
    sum = 0
    for i in x:
        sum += i * i
    product = 1
    for j in range(len(x)):
        product *= np.cos(x[j] / np.sqrt(j + 1))
    return 1 + sum / 4000 - product

#Rastrigin's Function
def f8(x):
    sum = 0.0
    for i in range(0, len(x)):
        sum += (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10)
    return sum

#Katsuura Function
def f9(x):
    D = len(x)
    P = 1
    S = 0

    for i in range(0, D):
        for j in range(1, 33):
            S = S + (abs(((2 ** j) * x[i]) - np.round((2 ** j) * x[i])) / 2 ** j)
        P = P * (1 + i * S) ** (10 / (D ** 12))

    prod = ((10 / (D ** 2)) * P) - (10 / D ** 2)

    return(prod)




#plotting

def plot(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # set the 3d axes
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm )
    plt.show()

def plotall():
    X = np.linspace(-100, 100, 100)  # points from 0..10 in the x axis
    Y = np.linspace(-100, 100, 100)  # points from 0..10 in the y axis
    X, Y = np.meshgrid(X, Y)  # create meshgrid

    #function 1
    Z = f1([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 2
    Z = f2([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 3
    Z = f3([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 4
    Z = f4([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 5
    Z = f5([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 6
    Z = f6([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 7
    Z = f7([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 8
    Z = f8([X, Y])  # Calculate Z
    plot(X,Y,Z)

    # Function 9
    Z = f9([X, Y])  # Calculate Z
    plot(X,Y,Z)

#plot the results
def plotRes(p,d):
    c = []
    for i in range (0,len(p)):
        c.append(i)
    plt.plot(c,p,'b-')
    plt.plot(c,d,'r-')
    plt.suptitle('Average Best Cost over 51 runs',fontsize=20)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Cost', fontsize=16)
    plt.show()


class Particle:

    def __init__(self,D):
        self.pos = np.random.uniform(-10,10,size=D)
        #initialize velocity as 0
        self.vel = np.random.uniform(0,0,size=D)
        #initialize cost for first position
        self.cost = f(self.pos)
        self.bCost = self.cost
        self.bPos = self.pos

    def updatePos(self,x):
        self.pos = x

    def updateCost(self,x):
        self.cost = f(x)

    def updateBC(self,x):
        self.bCost = x

    def updateBP(self,x):
        self.bPos = x

    def updateVel(self,x):
        self.vel = x


#problem definition
def pso(x):
    D=x       #dimension size
    runs = 51
    nPop = 100    #population size
    maxIt = int(np.round((3000*D)/nPop)) #number of iterations
    C1 = 2.05
    C2 = 2.05
    W = 0.9
    count = 0
    fRes = []
    bRes = []
    wRes = []
    globalPos = []


    #fill fRres with 0s
    for i in range(0, maxIt):
        fRes.append(0)

    #code to execute each run
    for z in range(0,runs):
        pop=[]
        for i in range(0,nPop):         #initialize population array and fill
            pop.append(Particle(D))

        #initialize global bests

        gbPos = pop[0].pos
        gbCost = pop[0].cost
        for i in range(0,nPop):
            if pop[i].cost < gbCost:
                gbPos = pop[i].pos
                gbCost = pop[i].cost

        #initialize results array and fill with 0s
        results = []
        for i in range(0,maxIt):
            results.append(0)

        #Update everything
        for i in range(0,maxIt):
            W = ((0.4 - 0.9) / maxIt) * i + 0.9
            results[i] = gbCost
            for j in range(0,nPop):
                temp1 = np.random.rand(D)
                temp2 = np.random.rand(D)
                newVel = (W * pop[j].vel)\
                             + (C1 * temp1 * (pop[j].bPos - pop[j].pos))\
                             + (C2 * temp2 * (gbPos - pop[j].pos))
                pop[j].updateVel(newVel)
                newPos = pop[j].pos + pop[j].vel
                for x in range(0,len(newPos)):
                    if newPos[x] > 10 or newPos[x] < -10:
                        break
                    else:
                        pop[j].updatePos(newPos)
                pop[j].updateCost(pop[j].pos)
                if i == 0 and j == 0:
                    gbPos = pop[j].pos
                    gbCost = pop[j].cost

                if pop[j].cost < pop[j].bCost:
                    pop[j].updateBC(pop[j].cost)
                    pop[j].updateBP(pop[j].pos)
                    if pop[j].cost < gbCost:
                        gbPos = pop[j].pos
                        gbCost = pop[j].cost

            #fill results array


        for i in range(0,len(fRes)):
            fRes[i] = fRes[i] + results[i]
            count = count + 1
        globalPos.append(gbPos)
        bRes.append(min(results))
        wRes.append(max(results))
        #pRes.append(gb)
    for i in range(0, len(fRes)):
        fRes[i] = fRes[i] / count
    temp = bRes[0]
    tempi = 0
    for i in range(1,len(bRes)):
        if bRes[i] < temp:
            temp = bRes[i]
            tempi = i

    return fRes, bRes, wRes, temp, globalPos[tempi]

def de(x):
    D = x
    nPop = 100
    Cr = 0.9
    F = 0.8
    maxIt = int(np.round((3000 * D) / nPop))
    #maxIt = 30 * D
    fRes = []
    bRes = []
    wRes = []
    pRes = []
    count = 0
    runs = 51

    for i in range(0, maxIt):
        fRes.append(0)

    for z in range(0,runs):
        results = []
        for i in range(0, maxIt):
            results.append(0)
        # initialize population array and fill
        pop = []
        for i in range(0, nPop):
            pop.append(Particle(D))

        ePop = pop
        #for i in range(0, nPop):
            #ePop.append(pop[i].pos)
        temp = []
        for i in range(0,nPop):
            temp.append(pop[i].cost)

        gb = Particle(D)
        gb.updatePos(pop[0].pos)
        gb.updateCost(pop[0].pos)
        # gbPos = pop[0].pos
        # gbCost = pop[0].cost
        for i in range(0, nPop):
            if pop[i].cost < gb.cost:
                gb.updatePos(pop[i].pos)
                gb.updateCost(pop[i].pos)
                # gbPos = pop[i].pos
                # gbCost = pop[i].cost
        for q in range(0, maxIt):
            cost = []
            xN = []
            results[q] = gb.cost
            for i in range(0, nPop):

                x = ePop[i].pos
                a = 0
                b = 0
                c = 0
                # MUTATION
                while i == a or i == b or i == c or a == b or a == c or b == c:
                    a = random.randint(0, nPop - 1)
                    b = random.randint(0, nPop - 1)
                    c = random.randint(0, nPop - 1)
                xa = ePop[a].pos
                xb = ePop[b].pos
                xc = ePop[c].pos
                Vx = [xa[k] - xb[k] for k in range(0, D)]
                Vx = [Vx[k] * F for k in range(len(Vx))]
                Vx = [Vx[k] + xc[k] for k in range(0, D)]

                # CROSSOVER
                Ux = np.empty((D))
                for j in list(range(0, D)):
                    if (random.uniform(0, 1) < Cr):
                        Ux[j] = Vx[j]
                    else:
                        Ux[j] = x[j]

                # SELECTION
                fu = f(Ux)
                fx = f(x)
                if fu < fx:
                    xN.append(Ux)
                    cost.append(f(Ux))
                else:
                    xN.append(x)
                    cost.append(f(x))
            for i in range(0, len(ePop)):
                ePop[i].updatePos(xN[i])
                ePop[i].updateCost(xN[i])

            for i in range(0, len(ePop)):
                if pop[i].cost < gb.cost:
                    gb.updatePos(ePop[i].pos)
                    gb.updateCost(ePop[i].pos)
                    # gbPos = pop[i].pos
                    # gbCost = pop[i].cost


        pRes.append(gb)
        for i in range(0,len(fRes)):
            fRes[i] = fRes[i] + results[i]
            count = count + 1
        bRes.append(min(results))
        wRes.append(max(results))

    for i in range(0,len(fRes)):
        fRes[i] = fRes[i]/count
    temp = pRes[0]
    for i in range(0,len(pRes)):
        if pRes[i].cost < temp.cost:
            temp = pRes[i]


    return fRes, bRes, wRes, temp


#change dimension(2,5 or 10)
D = 2
deRes, deBest, deWorst, deSol = de(D)
print("DE best fitness, and solution")
print(deSol.cost)
print(deSol.pos)
print(" ")
print("DE best solution for each run")
print(deBest)
print(" ")
print("mean of best solutions")
print(np.mean(deBest))
print(" ")
print("standard deviation")
print(np.std(deBest))
print("------------------------------------------------------------------------------------------------")
psoRes, psoBest, psoWorst, psoFitness, psoSolution = pso(D)
print("PSO best fitness, and solution")
print(psoFitness)
print(psoSolution)
print(" ")
print("PSO best solution for each run")
print(psoBest)
print(" ")
print("mean of best solutions")
print(np.mean(psoBest))
print(" ")
print("standard deviation")
print(np.std(psoBest))


plotRes(psoRes,deRes)









