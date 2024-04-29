import cv2
import numpy as np
from matplotlib import pyplot as plt



def histogram_equalization(image):
    # 获取图像的高度和宽度
    height, width = image.shape

    # 计算直方图
    histogram = np.zeros(256, dtype=int)
    for i in range(height):
        for j in range(width):
            histogram[image[i, j]] += 1

    # 计算累计直方图
    cumulative_histogram = np.zeros(256, dtype=int)
    cumulative_histogram[0] = histogram[0]
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]

    # 进行均衡化
    equalized_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            equalized_image[i, j] = round((cumulative_histogram[image[i, j]] * 255) / (height * width))

    return equalized_image
#直方图均衡化
def equalizeHist(img):#传入单一维度的图像
    # s=f(img)
    # f变换为一个概率密度函数的积分变换
    Hist=np.zeros(256)
    F=np.zeros(256)
    W,H=np.shape(img)
    for i in range(W):
        for j in range(H):
            f=img[i,j]
            Hist[f]+=1
    sum=0
    for i in range(256):
        sum=sum+Hist[i]
        F[i]=sum
    F = np.round(F / (W * H)*255)
    new_img=np.zeros([W,H])
    for i in range(W):
        for j in range(H):
            new_img[i,j]=F[img[i,j]]
    # cv2.imshow('123',new_img)
    new_img = new_img.astype(np.uint8)
    return new_img
# img=plt.imread('lenna.png')
img=cv2.imread('123.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转化为灰度
(b,g,r)=cv2.split(img)
bF=equalizeHist(b)
gF=equalizeHist(g)
rF=equalizeHist(r)

new_img=cv2.merge((bF,gF,rF))#通道合成




# print(img)
# plt.figure(1)
# plt.imshow(img)
# plt.show()
# cv2.imshow('ewq',img)
# img=equalizeHist(img)

cv2.imshow('qwe',new_img)#必须要转化为8位无整形！！！！
cv2.waitKey()
# 计算数据的范围和频率分布
# cdf=data.cumsum()
# bins = np.arange(0, 256, 1)
# frequencies ,b= np.histogram(img, bins=bins)

# plt.bar(b,data,) # bin_edges的长度是hist长度 + 1 故舍弃bin_edges数组最后一个数值
# plt.plot(data)
# 绘制直方图
# img=np.reshape(img,-1)
# plt.hist(img, bins=256, rwidth=1)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Data')
# plt.show()
cv2.imwrite('out1.jpg',new_img)
#sobel边缘检测