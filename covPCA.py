from PCA_data import PCA_data
import numpy as np 
class PCA_cov(object):
    def __init__(self,n_components=20):
        self.n_components = n_components
        if self.n_components < 2:
            raise ValueError ("must specifiy n_components for CCIPCA")
    
    
    def Remove_arrange_data_mat(self,X):#移去均值
        self.mean = X.mean(axis=1)
        X= X -self.mean
        return 

    def Cx_generate(self,X): #生成协方差矩阵
        """self.Cx = np.matrix(np.zeros((self.n_sample,self.n_sample)))
        for i in range(self.n_feature):
            temp = X[:,i]*X[:,i].T
            self.Cx += temp
        self.Cx /= self.n_feature"""
        self.Cx = np.cov(X)
        return 
    
    def fit(self,X): #计算特征值
        self.n_sample,self.n_feature = X.shape
        self.Remove_arrange_data_mat(X)
        self.Cx_generate(X)
        feature_val,feature_vectors = np.linalg.eig(self.Cx)
        order = np.argsort(-feature_val)
        self.feature_vectors = np.matrix(np.zeros((self.n_sample,self.n_components)))
        for i in order[:self.n_components]:
            temp = feature_vectors[:,i]
            self.feature_vectors[:,i] = temp.copy().reshape((400, 1))
    
    def transform(self,X):   #获得压缩数据
        self.compass_data = (X-self.mean).T*self.feature_vectors
        return self.compass_data
    
    def predict(self):  #解压缩
        return (self.compass_data*self.feature_vectors.T).T+self.mean      


if __name__ == '__main__':
    a = PCA_data()
    imag_data = np.matrix(a.Read_face().astype('float64'))
    b = PCA_cov(2)
    b.fit(imag_data)
    compass_imag = b.transform(imag_data)
    decompass_imag = b.predict()
    a.picture_show(decompass_imag,'covPCA_2')
    #a.Imag_produce(decompass_imag,Dict='face_cov_new_test\\')   