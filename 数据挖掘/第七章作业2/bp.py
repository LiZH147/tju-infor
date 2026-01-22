import numpy as np
import matplotlib.pyplot as plt
import random

# 激活函数sigmod(x) 以及导数DS
def sigmoid(x):
    return np.tanh(x)
def DS(x):
    return 1 - (sigmoid(x) ** 2)

# 生成区间[a,b]内的随机数
def random_number(a,b):
    return (b-a)*random.random()+a 

# 生成一个m*n矩阵，并且设置默认零矩阵
def makematrix(m,n,fill=0.0):
    a = []
    for i in range(m):
        a.append([fill]*n)    # [fill]*3==[fill fill fill]
    return np.array(a)

# 构造三层BP神经网络
class BP:
    def __init__(self, num_in, num_hidden, num_out):
        self.num_in=num_in+1            # 输入层结点数 并增加一个偏置结点(阈值)
        self.num_hidden=num_hidden+1    # 隐藏层结点数 并增加一个偏置结点(阈值)
        self.num_out=num_out            # 输出层结点数
        
        self.active_in = np.array([-1.0]*self.num_in)
        self.active_hidden = np.array([-1.0]*self.num_hidden)
        self.active_out = np.array([-1.0]*self.num_out)

        self.weight_in=makematrix(self.num_in,self.num_hidden)      # in*hidden 的0矩阵
        self.weight_out=makematrix(self.num_hidden,self.num_out)    # hidden*out的0矩阵

        # 对权重矩阵weight赋初值
        for i in range(self.num_in):        
            for j in range(self.num_hidden):
                self.weight_in[i][j]=random_number(0.1,0.1)
        for i in range(self.num_hidden):    
            for j in range(self.num_out):
                self.weight_out[i][j]=random_number(0.1,0.1)
        # 偏差
        for j in range(self.num_hidden):
            self.weight_in[0][j]=0.1
        for j in range(self.num_out):
            self.weight_out[0][j]=0.1
        
        # 建立动量因子
        self.ci=makematrix(self.num_in,self.num_hidden)  
        self.co=makematrix(self.num_hidden,self.num_out)
 
    # 正向传播
    def update(self,inputs):
        if len(inputs)!=(self.num_in-1):
            raise ValueError("与输入层结点数不符")
        
        self.active_in[1:self.num_in]=inputs

        self.sum_hidden=np.dot(self.weight_in.T,self.active_in.reshape(-1,1))   # 叉乘
            # .T操作是对于array操作后的数组进行转置操作
            # .reshape(x,y)操作是对于array操作后的数组进行重新排列成一个x*y的矩阵，参数为负数表示无限制
        self.active_hidden=sigmoid(self.sum_hidden) # active_hidden[]作为输出层的输入数据
        self.active_hidden[0]=-1

        self.sum_out=np.dot(self.weight_out.T,self.active_hidden)
        self.active_out=sigmoid(self.sum_out)

        return self.active_out
    
    # 误差反向传播
    def errorbackpropagate(self,targets,lr,m): 
        if self.num_out==1:
            targets=[targets]
        if len(targets)!=self.num_out:
            raise ValueError("与输出层结点数不符")

        error=(1/2)*np.dot((targets.reshape(-1,1)-self.active_out).T,
                           (targets.reshape(-1,1)-self.active_out))
        
        # 输出层 误差信号
        self.error_out=(targets.reshape(-1,1)-self.active_out)*DS(self.sum_out) # DS(self.active_out)
        # 隐层 误差信号
        self.error_hidden=np.dot(self.weight_out,self.error_out)*DS(self.sum_hidden)    # DS(self.active_hidden)
 
        # 更新权值

        self.weight_out=self.weight_out+lr*np.dot(self.error_out,self.active_hidden.reshape(1,-1)).T+m*self.co
        self.co=lr*np.dot(self.error_out,self.active_hidden.reshape(1,-1)).T

        self.weight_in=self.weight_in+lr*np.dot(self.error_hidden,self.active_in.reshape(1,-1)).T+m*self.ci
        self.ci=lr*np.dot(self.error_hidden,self.active_in.reshape(1,-1)).T
 
        return error
 
    # 测试 用于作图
    def test(self,patterns):
        for i in patterns:
            print()
            # print(i[0:self.num_in-1],"->",self.update(i[0:self.num_in-1]))
        return self.update(i[0:self.num_in-1]) 
 
    # 权值
    def weights(self):
        print("输入层的权值：")
        print(self.weight_in)
        print("输出层的权值：")
        print(self.weight_out)
    
    def train(self,pattern,itera=100,lr=0.2,m=0.1):
        for i in range(itera):
            error=0.0   # 每一次迭代将error置0
            for j in pattern:  
                inputs=j[0:self.num_in-1]   # 根据输入层结点的个数确定传入结点值的个数
                targets=j[self.num_in-1:]   # 剩下的结点值作为输出层的值
                self.update(inputs)
                error=error+self.errorbackpropagate(targets,lr,m)   # 误差反向传播 计算总误差
            if i%10==0:
                print("########################误差 %-.5f ######################第%d次迭代" %(error, i))

X = list(np.arange(-1, 1.1, 0.1)) 
D = [-0.96, -0.577, -0.0729, 0.017, -0.641, -0.66, -0.11, 0.1336, -0.201, -0.434, -0.5, 
     -0.393, -0.1647, 0.0988, 0.3072,0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]
A = X + D  
patt = np.array([A] * 2)    

bp = BP(21, 13, 21)

bp.train(patt)

d = bp.test(patt)
# 查阅权重值
# bp.weights()
 
plt.plot(X, D, label="source data") 
plt.plot(X, d, label="predict data")  
plt.plot(X, D-d.flatten(), label="error data")  
plt.legend()
plt.show()
