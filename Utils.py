from asyncio.windows_events import NULL
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def printInfo(dataset, num_rig=20):
    print('\First ' + str(num_rig) + ' rows of the dataset:\t' + (dataset.dataframeName if
          hasattr(dataset, 'dataframeName') else '') + '\n')
    print(dataset.head(num_rig))
    print('\nDataset dimension:\t', dataset.shape)
    print('\n\nDataset description:\n')
    print(dataset.describe(include='all'))
    print('\n\nDataset info:')
    print(dataset.info())
    print('\n\n')

def control(dataset):
    print('\nDataset:\n')
    print(dataset.head())
    print('\n\nNumber of unique value per column:\n')
    print(dataset.isnull().sum())
    print('\nDuplicated row control:\n')
    print(dataset.duplicated().sum())
    print('\n')

def outlierDetection(data):
    mean = np.mean(data)
    std = np.std(data)
    print('mean of the dataset is', mean)
    print('std. deviation is', std)
    threshold = 3
    outlier = []
    for i in data:
        z = (i-mean)/std
        if z > threshold:
            outlier.append(i)
    print('outlier in dataset is', outlier)

def categPlotShow(dataset):
    riskColor = ['green', 'orange', 'red']
    riskOrder = ["low risk", "mid risk", "high risk"]
    plt.figure(figsize = (12,8))
    dataset["RiskLevel"].value_counts().plot(kind = "pie", labels = riskOrder, colors = riskColor, explode = [0.05, 0.05,  0.05], autopct='%1.1f%%', shadow = True)
    plt.title("Risk level distribution count")
    plt.show()

def numPlotShow(dataset, column):
    riskColor = ['green', 'orange', 'red']
    riskOrder = ["low risk", "mid risk", "high risk"]
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    sns.histplot(data = dataset, x = column, kde = True, ax = axs[0])
    sns.histplot(data = dataset, x = column, hue = "RiskLevel", kde = True, palette = riskColor, ax = axs[1])
    axs[0].set_title(f"{column} distribution histogram count")
    axs[1].set_title(f"{column} distribution histogram based on Risk level")
    plt.show()
    plt.close()

def scatterPlotShow(dataset, column1, column2):
    riskColor = ['green', 'orange', 'red']
    riskOrder = ["low risk", "mid risk", "high risk"]
    plt.figure(figsize = (12,8))
    plt.title("Scatter plot")
    sns.scatterplot(data=dataset, x=column1, y=column2, hue="RiskLevel", palette=riskColor)
    plt.show()
    plt.close()
    
def corrHeatMapShow(dataset):
    correlation = dataset.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(correlation, annot = True, linewidth = 1.7)
    plt.title("Correlation Heatmap")
    plt.show()
    plt.close()

def normalizeData(xTrain, xTest):
    scaler = MinMaxScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform (xTest) 
    return xTrain, xTest

def standardizeData(xTrain, xTest):
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform (xTest)
    return xTrain, xTest

def reduceData(dataset,n):
    pca = PCA(n_components=n)
    data = pca.fit_transform(dataset)
    return data

def metodReportShow(mod,xTrain,xTest,yTrain,yTest):
    pred = mod.predict(xTest)
    plt.figure(figsize=(7, 7))
    sns.heatmap(confusion_matrix(yTest, pred), annot=True, cmap='Purples')
    plt.xlabel("Predicted data")
    plt.ylabel("Real data")
    print('Result:\n')
    print(classification_report(yTest, pred))
    print(f'Accuracy:',accuracy_score(yTest, pred)* 100 ,'%')
    plt.show()
    plt.close()

    #Con K-fold
    stratifiedCv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cvScores = cross_val_score(mod, xTrain, yTrain, cv=stratifiedCv, scoring="accuracy")
    print(f'Accuracy mean with K-Fold:',cvScores.mean()* 100 ,'%')
    
    plt.title('Accuracy plot with K-Fold')
    i = range(1,11)
    plt.plot(i, cvScores, color='blue', marker='D')
    plt.xticks(ticks=cvScores, labels=cvScores)
    plt.grid()
    plt.show()
    plt.close()

def metodSVC(params,xTrain,xTest,yTrain,yTest,analisiDati):
    svc = RandomizedSearchCV(SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0),params)
    svc.fit(xTrain, yTrain)
    if(analisiDati == "si"):
        print("\n\nBest hyperparameter:", svc.best_params_)
        metodReportShow(svc,xTrain,xTest,yTrain,yTest)
    return svc
    
def metodRF(params,xTrain,xTest,yTrain,yTest,analisiDati):
    rf = RandomizedSearchCV(RandomForestClassifier(), params)
    rf.fit(xTrain, yTrain)
    if(analisiDati == "si"):
        print("Best hyperparameter:", rf.best_params_)
        metodReportShow(rf,xTrain,xTest,yTrain,yTest)
    return rf
    
def metodDT(params,xTrain,xTest,yTrain,yTest,analisiDati):
    dt = RandomizedSearchCV(DecisionTreeClassifier(), params)
    dt.fit(xTrain, yTrain)
    if(analisiDati == "si"):
        print("Best hyperparameter:", dt.best_params_)
        metodReportShow(dt,xTrain,xTest,yTrain,yTest)
    return dt
    
def metodKNN(xTrain,xTest,yTrain,yTest,analisiDati):
    maxAccuracy = 0
    bestK = 0
    for i in range(2,50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrain, yTrain)
        pred = knn.predict(xTest)
        if maxAccuracy < (accuracy_score(yTest, pred)):
            maxAccuracy = (accuracy_score(yTest, pred))
            bestK = i
    #Trovato il migliore K stampo le metriche per quello
    knn = KNeighborsClassifier(n_neighbors=bestK)
    knn.fit(xTrain, yTrain)
    if(analisiDati == "si"):
        print(f'Best K:',bestK)
        metodReportShow(knn,xTrain,xTest,yTrain,yTest)
    return knn

def metodNB(xTrain,xTest,yTrain,yTest,analisiDati):
    nb = GaussianNB()
    nb.fit(xTrain, yTrain)
    if(analisiDati == "si"):
        metodReportShow(nb,xTrain,xTest,yTrain,yTest)
    return nb
    

def metodKMeans(data,target):
    WCSS = []
    for i in range(1,11):
        model = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
        model.fit(data)
        WCSS.append(model.inertia_)
    fig = plt.figure(figsize = (13,8))
    plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')
    plt.xticks(np.arange(11))
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()
 
    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    for i in [2, 3, 4, 5]:
        '''
        Create KMeans instance for different number of clusters
        '''
        kmSilhouette = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(kmSilhouette, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(data) 
        
    #Per il metodo del gomito si è deciso di usare k=4 ma la silouette risuta leggermente migliore a 3
    km = KMeans(n_clusters=3, init = "k-means++", max_iter = 300, random_state = 42)
    pred = km.fit_predict(data)
    score = silhouette_score(data, km.labels_, metric='euclidean')
    #print('Omogeneità  : ', homogeneity_score(target, pred))
    #print('Completezza : ', completeness_score(target, pred))
    #print('V_measure   : ', v_measure_score(target, pred))
    print('Silhouette Score: %.3f' % score)

    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[pred == 0,0],data[pred == 0,1],data[pred == 0,2], s = 40 , color = 'green', label = "cluster 0")
    ax.scatter(data[pred == 1,0],data[pred == 1,1],data[pred == 1,2], s = 40 , color = 'yellow', label = "cluster 1")
    ax.scatter(data[pred == 2,0],data[pred == 2,1],data[pred == 2,2], s = 40 , color = 'blue', label = "cluster 2")
    #ax.scatter(data[pred == 3,0],data[pred == 3,1],data[pred == 3,2], s = 40 , color = 'blue', label = "cluster 3")
    ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2], s = 100, c = "red", label = "centroids")
    ax.set_xlabel('X-->')
    ax.set_ylabel('Y-->')
    ax.set_zlabel('Z-->')
    ax.legend()
    plt.show()
    plt.close()

    