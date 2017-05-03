# -*- coding: utf-8 -*-
import time
import input_data
import numpy as np

# 定数を定義
# 中間層のニューロン数
NUM = 90
# 訓練データ数
TRAIN_DATA = 110000
# テストデータ数
TEST_DATA = 10000
# 学習率
n = 0.01
# ノイズの有無
NOISE_FLAG = True
# ノイズの大きさ
NOISE_SIZE = 0.25

# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# MNIST データセットのダウンロードと読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 入力層→中間層重みと閾値を表す変数を用意する (初期値は乱数-0.01~0.01とする)
w1 = np.random.rand(NUM,784)
w1 = w1 * 0.02 - 0.01
b1 = np.random.rand(NUM)
b1 = b1[:,np.newaxis]

# 中間層→出力層重みと閾値を表す変数を用意する (初期値は乱数-0.01~0.01とする)
w2 = np.random.rand(10,NUM)
w2 = w2 * 0.02 - 0.01
b2 = np.random.rand(10)
b2 = b2[:,np.newaxis]

# 学習スタート
start_time = time.time()

# 訓練データを使って学習
for i in range(0,TRAIN_DATA):  

    # MNISTのi番目の訓練データを配列に格納
    train_image = mnist.train.images[i%55000]
    train_image = train_image[:,np.newaxis]
    train_label = mnist.train.labels[i%55000]
    train_label = train_label[:,np.newaxis]

    # ノイズを加える
    if NOISE_FLAG:
        n_num = int(784*NOISE_SIZE)
        for p in range(0,n_num):
            x = np.random.randint(784)
            noise = 1.0 / 255.0 * np.random.randint(0,255)
            train_image[x][0] = noise

    # 中間層の値を計算
    x1 = w1.dot(train_image) + b1
    X1 = sigmoid(x1)
    
    # 出力層の値を計算
    x2 = w2.dot(X1) + b2
    X2 = sigmoid(x2)
    
    # 誤差計算
    E = train_label - X2
    error = np.sum(E*E)

    print 'error'
    print error
 
    # -- 誤差逆伝播 --

    # w2の誤差信号の計算
    d2 = -(train_label - X2) * X2 * (1 - X2)
    
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

    # ノイズを加える
    if NOISE_FLAG:
        n_num = int(784*NOISE_SIZE)
        for p in range(0,n_num):
            x = np.random.randint(784)
            noise = 1.0 / 255.0 * np.random.randint(0,255)
            test_image[x][0] = noise
            
    # 中間層の値を計算
    test_x1 = w1.dot(test_image) + b1
    test_X1 = sigmoid(test_x1)

    #print 'test_X1'
    #print test_X1
    
    # 出力層の値を計算
    test_x2 = w2.dot(test_X1) + b2
    test_X2 = sigmoid(test_x2)

    # 出力値から答えを決定
    max = 0
    ans = -1
    for j in range(0,10):
        if test_X2[j][0] > max:
            max = test_X2[j][0]
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
