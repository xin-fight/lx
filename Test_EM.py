import matplotlib.pyplot as plt
import numpy as np
import math
import random
MAX = 10000               #最大循环次数
MIN = 0.0001              #最小阈值

def Gaosi(y,u,sigma):
    sum = 0
    sum += (y - u)**2        #sum     为 (y-u)*(y-u)
    return math.exp(-((y - u)**2)/(2*sigma)) / (math.sqrt(2*math.pi)*math.sqrt(sigma))  #正态分布公式


def E_step(r,u1,u2,sigma1,sigma2,a1,a2):
    E_class1 = []  #进行的E后的分类
    E_class2 = []
    P1 = []
    P2 = []
    for i in range(len(r)):
        p1 = a1*Gaosi(r[i],u1,sigma1)/(a1*Gaosi(r[i],u1,sigma1)+a2*Gaosi(r[i],u2,sigma2))
        p2 = a2*Gaosi(r[i],u2,sigma2)/(a1*Gaosi(r[i],u1,sigma1)+a2*Gaosi(r[i],u2,sigma2))
        P1.append(p1)
        P2.append(p2)
        #print (p1,"    ",p2)
        #print (P[0][0],"  sadasdasdas",P[0][1])
    return  P1,P2

def M_step(r,P1,p2):
    ######### 更新新的u ########
    #print(len(r),"     sdsdfsd")
    SUMy1 = 0            #预期第一类所对应的高斯值的 和
    SUMy2 = 0
    Y1 = 0                #响应度
    Y2 = 0
    for i in range(len(r)):
        Y1 += P1[i]
        SUMy1 += P1[i]*r[i]
       #print(Y1,"  dsasdsaf")
    Enu1 = SUMy1/Y1

    for i in range(len(r)):
        SUMy2 += p2[i]*r[i]
        Y2 += P2[i]
    Enu2 = SUMy2/Y2

    ########## 更新新的sigma方 #########
    SUM1 = 0             #预期第一类对应的 (y-u)*(y-u)
    SUM2 = 0
    for i in range(len(r)):
        SUM1 += P1[i]*math.pow(r[i]-Enu1,2)
    Ensigma1 = SUM1/Y1

    for i in range(len(r)):
        SUM2 += P2[i]*math.pow(r[i]-Enu2,2)
    Ensigma2 = SUM2/Y2

    ########### 更新新的a ###########
    Ena1 = Y1/(len(r))
    Ena2 = Y2/(len(r))

    return Enu1,Enu2,Ensigma1,Ensigma2,Ena1,Ena2

def LOG(r,u1,u2,sigma1,sigma2,a1,a2,P1,P2):
    Y1 = 0
    Y2 = 0
    for i in range(len(r)):
        Y1 += P1[i]
        Y2 += P2[i]
    SUM1 = 0                #对数似然式子中后面 之和
    SUM2 = 0
    for i in range(len(r)):
        SUM1 += P1[i]*math.log(1/math.sqrt(2*math.pi))-math.log(math.sqrt(sigma1))-1/(2*sigma1)*math.pow(r[i]-u1,2)
    for i in range(len(r)):
        SUM2 += P2[i]*math.log(1/math.sqrt(2*math.pi))-math.log(math.sqrt(sigma2))-1/(2*sigma2)*math.pow(r[i]-u2,2)

    log = (Y1*math.log(a1)+SUM1)+(Y2*math.log(a2)+SUM2)
    return log

if __name__ == "__main__":
    u1 = 0
    u2 = 10
    sigma1 = 10          #该出sigma意味着    sigma平方
    sigma2 = 10
    r = []
    GS = 10000
    result1 = np.random.normal(u1, math.sqrt(sigma1), GS)
    result2 = np.random.normal(u2, math.sqrt(sigma2), GS)
    r.extend(result1)
    r.extend(result2)
    print(len(result1),"   ",len(result2),"    ",len(r))

    ####u1...为真实值       Eu1...为随机值     Ea[0] 对应a1 Ea[1]对应  a2
    #######随机初始化    ak  u1 u2 sigma1 sigma2
    Ea=[0,0]                #保存的ak
    Ea[0]=random.random()
    Ea[1]=1-Ea[0]
    Eu1 = random.random()
    Eu2 = random.uniform(u2-2,u2+2)
    Esigma1 = random.uniform(8,12)
    Esigma2 = random.uniform(8,12)
    print("初始值随机为：   a1: ", Ea[0], "      a2: ", Ea[1], "     u1: ", Eu1, "     u2: ", Eu2)
    ##########
    O_log = 0
    for M in range(MAX):
        P1,P2 = E_step(r,Eu1,Eu2,Esigma1,Esigma2,Ea[0],Ea[1])

        Eu1,Eu2,Esigma1,Esigma2,Ea[0],Ea[1] = M_step(r,P1,P2)

        log = LOG(r,Eu1,Eu2,Esigma1,Esigma2,Ea[0],Ea[1],P1,P2)
        if M%50 == 0:
            print("迭代次数",M,"     u1: ", Eu1, "     u2: ", Eu2,"       sigma1: ",Esigma1,"       sigma2: ",Esigma2,"       a1: ", Ea[0], "      a2: ", Ea[1])
            #print("每个类的个数 ",len(E_class1),"    ",len(E_class2))
            print("对数似然为： ",log,"    ",O_log,"     ",abs(log-O_log))

        if abs(log-O_log)<MIN:
            break
        else:
            O_log = log

    ######## 计算loss #######
    NN1 = 0
    NN2 = 0
    E_class1 = []
    E_class2 = []
    for i in range(len(r)):
        if (i<GS):
            if P1[i]>0.5:
                NN1+=1

        if (i>=GS):
            if P1[i]<=P2[i]:
                NN2+=1

    for i in range(len(r)):
        if P1[i]>=P2[i]:
            E_class1.append(r[i])
        else:
            E_class2.append(r[i])
    loss = (NN2+NN1)/len(r)
    print ("loss: ",loss)
    plt.hist(result1,100)
    plt.hist(result2,100)
    plt.show()

    plt.hist(E_class1,100)
    plt.hist(E_class2,100)
    plt.show()

