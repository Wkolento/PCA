import numpy as np 
from PCA_data import PCA_data
from copy import deepcopy
def norm(data):
    for i in range(len(data)):
        data[i]/=np.linalg.norm(data[i])
class NW_PCA(object):
    def __init__(self,components = 20):
        self.n_components = components


    def fit(self,data,alpha = 0.03,max_iter=10):
        self.X = deepcopy(data)
        self.__n_samples, self.__n_features = self.X.shape 
        self.mean_ = self.X.mean(axis = 0)
        self.X = self.X - self.mean_              #移去均值
        self.W = np.random.rand(self.__n_samples,self.__n_samples)
        self.__features_fit(self.X,alpha,max_iter)

    def __features_fit(self, X,alpha,max_iter):
        alph = alpha
        for i in range(max_iter):
            for i in range(self.n_components):
                if i != 0:
                    e = e-np.outer(self.W[i-1],y)
                else:
                    e = deepcopy(self.X)
                norm(e)
                y = self.W[i].dot(e)
                #norm(y)            
                self.W[i] += alph*np.dot(y,(e-np.outer(self.W[i],y)).T)
                self.W[i] = self.W[i]/np.linalg.norm(self.W[i])
                #y = self.W[i].dot(self.X)
        self.components =np.zeros((self.n_components,self.__n_samples))
        for i in range(self.n_components):
            self.components[i] = self.W[i]
    
    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed.T, self.components.T)
        return X_transformed

    def predict(self, X):
        decompass_data = np.dot(X, self.components).T + self.mean_
        return decompass_data

if __name__ == '__main__':
                    #提取实现
    a = PCA_data()
    imag_data = a.Read_face().astype('float64')
    norm(imag_data)
    
    b = NW_PCA(40)    #修改主元数
    b.fit(imag_data,max_iter=100)   #修改迭代次数

    compass_data = b.transform(imag_data)
    decompass_imag = b.predict(compass_data)
    #a.Imag_produce(decompass_imag,Dict='face_nw\\')
    a.picture_show(decompass_imag,'oja-PCA_40')