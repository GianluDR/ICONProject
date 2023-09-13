import warnings 

from Utils import *
from sklearn.model_selection import train_test_split
from dataAnalysis import *

def trainModels(analisiDati):
    #print(analisiDati)
    datasetLink = 'MaternalHealthRisk.csv'
    warnings.filterwarnings('ignore')

    dataset = pd.read_csv(datasetLink)
    dataset.dataframeName = 'Maternity health risk dataset'

    #CAMBIO NOMI PER MAGGIORE LEGGIBILITA
    dataset.columns = ['Age', 'SystolicBloodPressure', 'DiastolicBloodPressure', 'BloodSugar', 'BodyTemp', 'HeartRate', 'RiskLevel']

    dataset['RiskLevel'] = dataset['RiskLevel'].map({'low risk':0,'mid risk':1,'high risk':2})

    #Attraverso l'analisi grafica dei dati e controllo nel dataset sono stati rilevati alcuni valori anomali in Heartrate. 
    #Ha alcune registrazioni di 7 battuti per minuto impossibili per un normale adulto anche a riposo. Eliminiamo quelle righe.
    dataset.drop(dataset[dataset.HeartRate == 7].index, inplace=True)

    #Dopo vari test  e analisi fra i dati si è deciso di droppare heartrate perchè non da nessuna informazione
    dataset = dataset.drop("HeartRate", axis=1)

    #dataset splitting
    x = dataset.drop("RiskLevel", axis=1)
    y = dataset['RiskLevel']

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)
    xTrainS, xTestS = standardizeData(xTrain, xTest)
    xTrainN, xTestN = normalizeData(xTrain, xTest)

    print("\nCaricamento dati... fra 2-3 minuti avrai il tuo risultato...")

    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    svcTrained = metodSVC(params,xTrain,xTest,yTrain,yTest,analisiDati)

    params = {'n_estimators': [25, 50, 75, 100, 150, 200, 250]}
    rfTrained = metodRF(params,xTrain,xTest,yTrain,yTest,analisiDati)

    params = {'criterion': ['gini', 'entropy', 'log_loss']}
    dtTrained = metodDT(params,xTrainS,xTestS,yTrain,yTest,analisiDati)

    knnTrained = metodKNN(xTrainN,xTestN,yTrain,yTest,analisiDati)

    nbTrained = metodNB(xTrain,xTest,yTrain,yTest,analisiDati)

    return svcTrained,rfTrained,dtTrained,knnTrained,nbTrained
