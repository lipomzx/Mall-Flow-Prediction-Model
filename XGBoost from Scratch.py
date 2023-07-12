import numpy as np
import pandas as pd
from math import e


class Node:

    '''
        With the internal gain to find the optimal split value uses both gradient and hessian. 

        Parameters:
            x: dataframe of the training data 
            gradient: negative gradient of the loss function 
            hessian: 2nd order derivative of the loss function 
            idxs: used to keep track of samples within the tree structure
            subsample_cols: denotes the fraction of observations to be randomly samples for each tree.
            min_leaf: minimum num of samples for a node to be a leaf
            min_child_weight: sum of hessian within current node (purity), overfitting 
            depth: limitation of depth of the tree
            lambda: l2 regularization on weights, control overfitting 
            gamma: Gamma specifies the minimum loss reduction required to make a split, control overfitting
            eps: 1/ sketch_steps num of bins

    '''

    def __init__(self, x, gradient, hessian, idxs, subsample_cols = 0.8, min_leaf= 5, min_child_weight = 1, depth=10, 
                 lambda_ = 1, gamma = 1, eps=0.1):
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols*self.col_count)]
        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])
        self.score = float('-inf')
        self.find_varsplit()

    
    def compute_gamma(self, gradient, hessian):
        return (-np.sum(gradient)/ (np.sum(hessian)+ self.lambda_))
    
    def find_varsplit(self):
        '''
        Scan through every column and calculates the best split point 
        After find the best split, 2 new child nodes are created and the depth of the tree increase by 1
        If no split is better than the score initailized at the beginning then nothing change 
        '''
        for c in self.column_subsample: 
            self.find_greedy_split(c)
        if self.is_leaf: return 
        x = self.split_col
        lb = np.nonzero(x <= self.split)[0]
        rb = np.nonzero(x > self.split)[0]
        self.lb = Node(x = self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[lb], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
        self.rb = Node(x = self.x, gradient = self.gradient, hessian = self.hessian, idxs = self.idxs[rb], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
   

    
    def find_greedy_split(self, var_idx):
        '''
        For the given feature, greedily calculates the gain at each split
        Globally updates the best score and split point if exists better split point 
        '''

        x = self.x[self.idxs, var_idx]
        for r in range(self.row_count):
            left = x <= x[r]
            right = x > x[r]

            lf_idx = np.nonzero(x <=x[r])[0]
            ri_idx = np.nonzero(x > x[r])[0]
            if(right.sum() < self.min_leaf or left.sum() < self.min_leaf
               or self.hessian[lf_idx].sum() < self.min_child_weight
               or self.hessian[ri_idx].sum() < self.min_child_weight): continue
            
            score = self.gain(left, right)
            if score > self.score_:
                self.var_idx = var_idx
                self.score_ = score
                self.split = x[r]

    def weighted_quantile_sketch(self, var_idx):
        '''
        approximation to the exact greedy approach faster for bigger datasets where isn't feasible to 
        calculate the gain at every split point.
        '''
        x = self.x[self.idxs, var_idx]
        hessian_ = self.hessian[self.idx]
        df = pd.DataFrame({'feature':x, 'hess':hessian_})

        df.sort_values(by=['feature'], ascending=True, inplace=True)
        hess_sum = df['hess'].sum()
        df['rank'] = df.apply(lambda x: (1/hess_sum) * sum(df[df['feature'] < x['feature']]['hess']), axis=1)

        for row in range(df.shape[0]-1):
            rk_sk_j, rk_sk_1 = df['rank'].iloc[row:row+2]
            diff = abs(rk_sk_j - rk_sk_1)
            if (diff >= self.eps):
                continue

            split_value = (df['rank'].iloc(row+1) + df['rank'].iloc[row]) / 2
            lhs = x <=split_value
            rhs = x > split_value

            l_idx = np.nonzero(x <= split_value)[0]
            r_idx = np.nonzero(x > split_value)[0]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf
               or self.hessian[l_idx].sum() < self.min_child_weight
               or self.hessian[r_idx].sum() < self.min_child_weight): continue
            
            score = self.gain(lhs, rhs)
            if score > self.score_:
                self.var_idx = var_idx
                self.score_ = score
                self.split = split_value 

    def gain(self, lhs, rhs):
        '''
        Calculate the gain at a particular split point
        '''

        gradient = self.gradient[self.idxs]
        hessian = self.hessian[self.idxs]

        l_gradient = gradient[lhs].sum()
        l_hessian = hessian[lhs].sum()

        r_gradient = gradient[rhs].sum()
        r_hessian = hessian[rhs].sum()

        gain = 0.5 *((l_gradient**2/ (l_hessian+ self.lambda_)) + (r_gradient**2 / (r_hessian + self.lambda_)) 
                     - ((l_gradient+r_gradient)**2 / (l_hessian + r_hessian + self.lambda_))) - self.gamma
        
        return gain 

    @property
    # for oop without getter and setter, property(fget=None, fset=None, fdel=None, doc=None)
    
    ''' 
    temperature = property(get_temperature, set_temperature)
    above same as:
        @property 
        def temperature(self):
        return self._temperature

        @temperature.setter
        def temperature(self, temp):
        self._temperature = temp
    '''

    def is_leaf(self):
        return self.score == float('-inf') or self.depth <= 0 

    @property
    def split_col(self):
        return self.x[self.idxs , self.var_idx]
    
    def predict(self, x):
        return np.array([self.predict_row(x_i) for x_i in x])
    
    def predict_row(self, x_i):
        if self.is_leaf:
            return self.val

        node = self.lb if x_i[self.var_idx] <= self.split else self.rb 
        return node.predict_row(x_i)



class XGBoostTree:
    '''
    Wrapper class that provides a scikit learn interface to the recursive regression tree
    '''

    def fit(self, x, gradient, hessian, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
        self.dtree = Node(x, gradient, hessian, np.array(np.arange(len(x))), subsample_cols, min_leaf, min_child_weight, depth, lambda_, gamma, eps)
        return self

    def predict(self, x):
        return self.dtree.predict(x)
    

class XGBoostClassifier:
    '''
    Application of xgboost for binary classification 
    '''

    def __init__(self):
        self.estimators=[]

    @staticmethod
    def sigmoid(x):
        return 1/ (1 + np.exp(-x))
    
    # 1st order gradient log loss
    def grad(self, preds, labels):
        preds = self.sigmoid(preds)
        return (preds - labels)
    

    # 2nd order gradient logloss 
    def hess(self, preds, labels):
        preds = self.sigmoid(preds)
        return (preds * (1-preds))
    
    @staticmethod
    def log_odds(column):
        binary_yes = np.count_nonzero(column == 1)
        binary_no = np.count_nonzero(column == 0)
        return (np.log(binary_yes/binary_no))
    
    def fit(self, X, y, subsample_cols = 0.8,  min_child_weight = 1, depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, eps = 0.1):
        self.X = X, self.y = y
        self.depth = depth 
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma

        self.base_pred = np.full((X.shape[0], 1), 1).flatten().astype('float64')

        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth=self.depth, min_leaf=self.min_leaf, lambda_=self.lambda_, gamma=self.gamma, eps=self.eps, min_child_weight=self.min_child_weight, subsample_cols=self.subsample_cols)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)
        

    def predict_prob(self, X):
        pred = np.zeros(X.shape[0])
        for esti in self.estimators:
            pred += self.learning_rate * esti.predict(X)
        
        return (self.sigmoid(np.full(X.shape[0],1), 1).faltten().astype('float64') + pred)
    
    def predict(self, X):   
        pred = np.zeros(X.shape[0])
        for esti in self.estimators:
            pred += self.learning_rate * esti.predict(X)
        
        predicted_prob =  self.sigmoid(np.full(X.shape[0],1), 1).faltten().astype('float64') + pred
        preds= np.where(predicted_prob > np.mean(predicted_prob), 1, 0)
        return preds
    

class XGBoostRegressor:
    '''
    Application of xgboost for regression 
    '''
    def __init__(self):
        self.estimators = []

    @staticmethod
    def grad(preds, labels):
        return (2*(preds - labels))

    @staticmethod
    def hess(preds, labels):
        ''' 
        hessian of mse = constant value of two
        return an array of twos 
        https://math.stackexchange.com/questions/2581593/finding-hessian-of-linear-mse-using-index-notatio 
        '''
        return (np.full((preds.shape[0],1), 2).flatten().astype('float64'))
    
    def fit(self, X, y, subsample_cols = 0.8 , min_child_weight = 1, depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, eps = 0.1):
        self.X, self.y = X, y
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma
    
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten().astype('float64')

        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth = self.depth, min_leaf = self.min_leaf, lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps, min_child_weight = self.min_child_weight, subsample_cols = self.subsample_cols)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)



    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for est in self.estimators:
            pred += self.learning_rate * est.predict(X)

        return np.full((X.shape[0], 1), np.mean(self.y)).flatten().astype('float64') + pred
    





