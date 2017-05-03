# -*- coding: utf-8 -*-

import time
import input_data
import numpy as np

# 定数を定義
# 中間層1,2のニューロン数
NUM1 = 180
NUM2 = 45
# 訓練データ数
TRAIN_DATA = 3300000
# テストデータ数
TEST_DATA = 10000
# 学習率
n = 0.03

# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# MNIST データセットのダウンロードと読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 入力層→中間層1重みと閾値を表す変数を用意する (初期値は乱数-0.01~0.01とする)
w1 = np.random.rand(NUM1,784)
w1 = w1 * 0.02 - 0.01
b1 = np.random.rand(NUM1)
b1 = b1[:,np.newaxis]

# 中間層1→中間層2重みと閾値を表す変数を用意する (初期値は乱数-0.01~0.01とする)
w2 = np.random.rand(NUM2,NUM1)
w2 = w2 * 0.02 - 0.01
b2 = np.random.rand(NUM2)
b2 = b2[:,np.newaxis]

# 中間層→出力層重みと閾値を表す変数を用意する (初期値は乱数-0.01~0.01とする)
w3 = np.random.rand(10,NUM2)
w3 = w3 * 0.02 - 0.01
b3 = np.random.rand(10)
b3 = b3[:,np.newaxis]

# 学習スタート
start_time = time.time()

# 訓練データを使って学習
for i in range(0,TRAIN_DATA):  

    # MNISTのi番目の訓練データを配列に格納
    train_image = mnist.train.images[i%55000]
    train_image = train_image[:,np.newaxis]
    train_label = mnist.train.labels[i%55000]
    train_label = train_label[:,np.newaxis]

    # 中間層1の値を計算
    x1 = w1.dot(train_image) + b1
    X1 = sigmoid(x1)
    
    # 中間層2の値を計算
    x2 = w2.dot(X1) + b2
    X2 = sigmoid(x2)

    # 出力層の値を計算
    x3 = w3.dot(X2) + b3
    X3 = sigmoid(x3)
    
    # 誤差計算
    E = train_label - X3
    error = np.sum(E*E)

    print 'error (%d/%d)' % (i, TRAIN_DATA)
    print error

    # -- 誤差逆伝播 --

    # w3の誤差信号の計算
    d3 = -(train_label - X3) * X3 * (1 - X3)
    # w3の更新
    dw3 = -n * (d3.dot(X2.T)) 
    w3 = w3 + dw3
    # b3の更新
    db3 = -n * d3
    b3 = b3 + db3

    # w2の誤差信号の計算
    d2 = (w3.T).dot(d3) * X2 * (1 - X2)
    # w2の更新
    dw2 = -n * (d2.dot(X1.T)) 
    w2 = w2 + dw2
    # b2の更新
    db2 = -n * d2
    b2 = b2 + db2
    
    # w1の誤差信号の計算
    d1 = (w2.T).dot(d2) * X1 * (1 - X1) 
    # w1の更新
    dw1 = -n * (d1.dot(train_image.T))
    w1 = w1 + dw1
    # b1の更新
    db1 = -n * d1
    b1 = b1 + db1

# 学習終了
end_time = time.time()

# 正答率を出すための変数を用意
correct_number = 0
problem_number = 0

# テストデータを使って正答率の計算
for i in range(0,TEST_DATA):

    # MNISTのi番目のテストデータを配列に格納
    test_image = mnist.test.images[i]
    test_image = test_image[:,np.newaxis]
    test_label = mnist.test.labels[i]
    test_label = test_label[:,np.newaxis]

    # 中間層1の値を計算
    test_x1 = w1.dot(test_image) + b1
    test_X1 = sigmoid(test_x1)

    # 中間層2の値を計算
    test_x2 = w2.dot(test_X1) + b2
    test_X2 = sigmoid(test_x2)
    
    # 出力層の値を計算
    test_x3 = w3.dot(test_X2) + b3
    test_X3 = sigmoid(test_x3)

    # 出力値から答えを決定
    max = 0
    ans = -1
    for j in range(0,10):
        if test_X3[j][0] > max:
            max = test_X3[j][0]
            ans = j

    # 正しい答えをMNISTから取得
    correct_ans = -2
    for k in range(0,10):
        if test_label[k][0] == 1:
            correct_ans = k

    # 両者の答えを比較
    if ans == correct_ans:
        correct_number += 1

    problem_number += 1

rate = 1.0 * correct_number / problem_number

print 'Rate : %f' % rate
print "Learning Time : " + str(end_time - start_time)
