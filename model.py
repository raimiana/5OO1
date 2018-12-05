import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve

class Model():
    dataset = None
    train_features = None
    train_labels = None
    test_features = None

    def loadData(self,labelName):
        self.dataset = pd.read_csv('./train.csv', header = 0)
        self.test_features = pd.read_csv('./test.csv', header = 0)
        self.train_labels = self.dataset[labelName].to_frame()
        colLen = len(self.dataset.columns)
        self.train_features = self.dataset.iloc[:,:(colLen-1)]


    def preprData(self):
        combine = {'train': self.train_features, 'test': self.test_features}
        for key in combine.keys():
            dum = pd.get_dummies(combine[key]['penalty'], prefix='pena')
            combine[key] = combine[key].drop('penalty', axis=1)
            combine[key] = pd.concat([combine[key], dum], axis=1)
            combine[key]['pena_l1'] = combine[key].apply(lambda x: x['l1_ratio'] if x['pena_elasticnet']==1 else x['pena_l1'], axis=1)
            combine[key]['pena_l2'] = combine[key].apply(lambda x: 1-x['pena_l1'], axis=1)
            combine[key]['n_jobs'] = combine[key]['n_jobs'].apply(lambda v: 16 if v == -1 else v)
            combine[key]['n_cluster'] = combine[key]['n_classes']*combine[key]['n_clusters_per_class']
            combine[key]['n_num'] = combine[key]['n_samples']*combine[key]['n_features']
            # Delete attributes
            drop_attr = ['id','pena_elasticnet','l1_ratio']
            combine[key] = combine[key].drop(drop_attr, axis=1)


        self.train_features = combine['train']
        self.test_features = combine['test']

        print(self.train_features.head())
        # print(self.train_features.info())
        self.train_features.to_csv(r'feature.csv', index=False)

    def model_predict(self,alg):
        X = self.test_features.values
        data = alg.predict(X)
        ind = range(len(data))
        result = pd.DataFrame({'id': ind, 'time': data})
       #result.to_csv(r'result2.csv', float_format='%.16f', index=False)
        result['time'] = result['time'].apply(lambda t: 0 if t < 0 else t)
        result.to_csv('result.csv', float_format='%.16f', index=False)



    def trainModel(self):
        x_train = self.train_features.values
        y_train = self.train_labels.values.ravel()



        gbClf = GradientBoostingRegressor(
            learning_rate=0.005,
            n_estimators=3000,
            max_depth=12,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features='sqrt', subsample=0.65,random_state=10
        )

        # self.learning_plot(gbClf)

        gbClf.fit(x_train,y_train)
        score = cross_val_score(gbClf,x_train,y_train,scoring='mean_squared_error',cv=5)
        print(score)
        print(np.mean(score))

        self.model_predict(gbClf)



m = Model()
print('---------------LOAD DATA---------------')
m.loadData('time')
print('---------------PREPROCESSING DATA---------------')
m.preprData()
print('---------------TRAINING---------------')
m.trainModel()

