import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 更新参数，训练模型
def train(x_train, y_train, epoch):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.68  # 初始学习率
    reg_rate = 0.00001  # 正则项系数
    bg2_sum = 0  # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim)  # 用于存放权重的梯度平方和
    losslist = {}
    mislist = {}

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            sig = 1 / (1 + np.exp(-y_pre))
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= num
        w_g /= num

        # adagrad
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        # 更新权重和偏置
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

        # 每训练3轮，输出一次在训练集上的正确率

        loss = 0
        acc = 0
        result = np.zeros(num)
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            sig = 1 / (1 + np.exp(-y_pre))
            if sig >= 0.5:
                result[j] = 1
            else:
                result[j] = 0

            if result[j] == y_train[j]:
                acc += 1.0
            loss += (-1) * (y_train[j] * np.log(sig + 1e-10) + (1 - y_train[j]) * np.log(1 - sig + 1e-10))
        losslist[i] = loss / num
        mislist[i] = 1 - acc / num
        if i % 3 == 0:
            print('after {} epochs, the loss on train data is:'.format(i), loss / num)
            print('after {} epochs, the acc on train data is:'.format(i), acc / num)
        
    y = [losslist[0],losslist[1],losslist[2],losslist[3],losslist[4],losslist[5],losslist[6],losslist[7],losslist[8],losslist[9],losslist[10],losslist[11],losslist[12],losslist[13],losslist[14],losslist[15],losslist[16],losslist[17],losslist[18],losslist[19],losslist[20],losslist[21],losslist[22],losslist[23],losslist[24],losslist[25],losslist[26],losslist[27],losslist[28],losslist[29]]
    x = np.linspace(0, 29, 30)
    z = [mislist[0],mislist[1],mislist[2],mislist[3],mislist[4],mislist[5],mislist[6],mislist[7],mislist[8],mislist[9],mislist[10],mislist[11],mislist[12],mislist[13],mislist[14],mislist[15],mislist[16],mislist[17],mislist[18],mislist[19],mislist[20],mislist[21],mislist[22],mislist[23],mislist[24],mislist[25],mislist[26],mislist[27],mislist[28],mislist[29]]
    plt.plot(x, y, ls="-", lw=2, label="train loss")
    plt.plot(x, z, ls="-", lw=2, label="misclassification rate")
    plt.legend()
    plt.show()
    return weights, bias


# 验证模型效果
def validate(x_val, y_val, weights, bias):
    num = 1000
    loss = 0
    acc = 0
    result = np.zeros(num)
    for j in range(num):
        y_pre = weights.dot(x_val[j, :]) + bias
        sig = 1 / (1 + np.exp(-y_pre))
        if sig >= 0.5:
            result[j] = 1
        else:
            result[j] = 0

        if result[j] == y_val[j]:
            acc += 1.0
        loss += (-1) * (y_val[j] * np.log(sig + 1e-10) + (1 - y_val[j]) * np.log(1 - sig + 1e-10))
    print('the test loss is:' , loss / num)
    return acc / num


def main():
    # 从csv中读取有用的信息
    df = pd.read_csv('spam_train.csv')
    # 空值填0
    df = df.fillna(0)
    # (4000, 59)
    array = np.array(df)
    # (4000, 57)
    x = array[:, 1:-1]
    # scale
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    # (4000, )
    y = array[:, -1]

    # 划分训练集与验证集
    x_train, x_val = x[0:2999, :], x[2999:3999, :]
    y_train, y_val = y[0:2999], y[2999:3999]

    epoch = 30  # 训练轮数
    # 开始训练
    w, b = train(x_train, y_train, epoch)
    # 在验证集上看效果
    acc = validate(x_val, y_val, w, b)
    print('The acc on val data is:', acc)


if __name__ == '__main__':
    main()