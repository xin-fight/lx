import numpy as np
X = np.array([[3,3,1],[4,3,1],[1,1,-1]])

def set(w1=0,w2=0,b=0,rate=1):
    return w1,w2,b,rate

def dell():
    w1,w2,b,rate = set()
    xunhuan = 0
    sum = 0
    while(xunhuan<3):
        xunhuan = 0
        for i in range(len(X)):
            Y = w1*X[i][0]+w2*X[i][1]+b
            if(Y*X[i][2]<=0):           #未能正确分类  更新
                xunhuan = 0
                w1 = w1+rate*X[i][2]*X[i][0]
                w2 = w2+rate*X[i][2]*X[i][1]
                b  = b+rate*X[i][2]
                sum+=1
                print("{0}+is  : {1}x +{2}y +{3}".format(sum,w1,w2,b)) #正则输出
            else:
                xunhuan +=1
    return 0

if __name__ == '__main__':
    dell()



