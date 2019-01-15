from PIL import Image
import numpy as np
import math 
import scipy
import os, sys
import matplotlib.pyplot as plt
class PCA_data:
    def __init__(self):
        self.ori_data = np.array([])
        self.imag_shape = (112,92)   #每个图片的像素分布
    def Read_face(self,dict='face\\s',x=0):
        #生成包含人脸数据的矩阵
        for i in range(1,41):
            for j in range(1,11):
                temp = i 
                Dictory = dict+str(temp)+ "\\"+str(j)+".bmp"
                im = Image.open(Dictory)
                imag = np.array([np.array(im).flatten()])
                if len(self.ori_data)==0:
                    self.ori_data = imag
                else :
                    self.ori_data = np.r_[self.ori_data,imag]
                del im,imag
        return  self.ori_data
    def Imag_produce(self,cov_data,Dict='face_restoration\\'):
        #将压缩后恢复的数据还原为图像
        length = len(cov_data)
        for i in range(length):
            dictory = Dict +str(int(i/10)+1)
            os.makedirs(dictory,exist_ok=True)
            bmp_data = np.array([cov_data[i]])
            new_im = Image.fromarray(bmp_data.reshape(self.imag_shape))
            new_im.save(dictory+'\\'+str(i%10+1)+'.bmp')
        return 

    def picture_show(self,pictures,title,order = 0):
        #numb = len(pictures)                     
                            #获取图像数目
        figure = plt.figure(1)      
                            #初始化
        for i in range(80):
            img = pictures[i*5+order].reshape(self.imag_shape)
                                        #图像还原
            plt.subplot(8,10,i+1)
            #vmax = max(pictures[i].max(), -pictures[i].min())                            #选择numb行，1列
            plt.imshow(img,cmap='gray',interpolation='nearest') #,vmin=-vmax, vmax=vmax
                                        #显示灰度图像
            plt.xticks(())
            plt.yticks(())
           # plt.imshow(img)
        figure.suptitle(title)          #标题
        #plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        figure.tight_layout()#调整整体空白
        plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
        plt.show()                      #显示

    def SNR(self,stand,restore):
        stand = stand.astype('float64')
        restore = restore.astype('float64')
        num = stand.shape[0]
        SNR_Sum = 0
        for i in range(num):
            up = np.sum(stand[i]**2)
            down = np.sum((restore[i]-stand[i])**2)
            SNR_Sum+=10*math.log(up/down,10)
        return  SNR_Sum/num

if __name__ == '__main__':
                #功能测试
    a = PCA_data()
    b = PCA_data()
    cov = a.Read_face()
    #a.picture_show(cov,'ori_data',4)
    res = b.Read_face(dict='face_cov\\')
    print('SNR=',a.SNR(cov,res))
    #print(cov)
    #a.Imag_produce(cov)