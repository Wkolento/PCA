from PCA_data import PCA_data
import numpy as np
from scipy import linalg as la
from sklearn.utils import check_array, as_float_array

class CCIPCA(object):
        
    def __init__(self, n_components=2, amnesic=2.0, copy=True):
        self.n_components = n_components
        if self.n_components < 2:
            raise ValueError ("must specifiy n_components for CCIPCA")
            
        self.copy = copy
        self.amnesic = amnesic
        self.iteration = 0

    def fit(self, X):        
        X = check_array(X)
        n_samples, n_features = X.shape 
        X = as_float_array(X, copy=self.copy)
        
        # init
        if self.iteration == 0:  
            self.mean_ = np.zeros([n_features], np.float)                           #行向量与行向量相加
            self.components_ = np.zeros([self.n_components,n_features], np.float)
        else:
            if n_features != self.components_.shape[1]:
                raise ValueError('The dimensionality of the new data and the existing components_ does not match')   
        
        # incrementally fit the model
        for i in range(0,n_samples):
            self.partial_fit(X[i,:])
        
        # update explained_variance_ratio_
        self.explained_variance_ratio_ = np.sqrt(np.sum(self.components_**2,axis=1))
        
        # sort by explained_variance_ratio_
        idx = np.argsort(-self.explained_variance_ratio_)
        self.explained_variance_ratio_ = self.explained_variance_ratio_[idx]
        self.components_ = self.components_[idx,:]
        
        # re-normalize
        self.explained_variance_ratio_ = (self.explained_variance_ratio_ / self.explained_variance_ratio_.sum())
           
        for r in range(0,self.components_.shape[0]):  #归一化
            self.components_[r,:] /= np.sqrt(np.dot(self.components_[r,:],self.components_[r,:]))
        #self.components_t = np.zeros((40,self.components_.shape[1]))
        return self
        
    def _amnestic(self, t):               # amnestic function
        if t <= int(self.amnesic):
            _rr = float(t+2-1)/float(t+2)    
            _lr = float(1)/float(t+2)    
        else:
            _rr = float(t+2-self.amnesic)/float(t+2)    
            _lr = float(1+self.amnesic)/float(t+2)
        
        return [_rr, _lr]
    

    def partial_fit(self, u):
        n = float(self.iteration)
        V = self.components_
        w1,w2=self._amnestic(n)
        self.mean_ = float(n+1-1)/float(n+1)*self.mean_ + float(1)/float(n+1)*u
        if n != 0:
        # mean center u        
            u = u - self.mean_
    
        # update components

        for j in range(0,self.n_components):
            
            if j > n:
                # the component has already been init to a zerovec
                pass
            
            elif j == n:
                # set the component to u 
                V[j,:] = u
                normedV = V[j,:] / la.norm(V[j,:])
            else:       
                # update the components
                V[j,:] = w1*V[j,:] + w2*np.dot(u,V[j,:])*u / la.norm(V[j,:])
                
                normedV = V[j,:] / la.norm(V[j,:])
            
            u = u - np.dot(np.dot(u.T,normedV),normedV)

        self.iteration += 1
        self.components_ = V
        return
    
    def transform(self, X):
        X = check_array(X)
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed

    def predict(self, X):
        decompass_data = np.dot(X, self.components_) + self.mean_
        return decompass_data


if __name__ == '__main__':
    a = PCA_data()
    imag_data = a.Read_face()
    
    b = CCIPCA(400)    
    b.fit(imag_data)

    compass_data = b.transform(imag_data)
    decompass_imag = b.predict(compass_data)
    a.picture_show(decompass_imag,'CCIPCA_400')
    #a.Imag_produce(decompass_imag)