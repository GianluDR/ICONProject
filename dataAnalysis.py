import warnings 

from Utils import *
from sklearn.model_selection import train_test_split

def dataShow():
    datasetLink = 'MaternalHealthRisk.csv'
    warnings.filterwarnings('ignore')

    dataset = pd.read_csv(datasetLink)
    dataset.dataframeName = 'Maternity health risk dataset'

    #CAMBIO NOMI PER MAGGIORE LEGGIBILITA
    dataset.columns = ['Age', 'SystolicBloodPressure', 'DiastolicBloodPressure', 'BloodSugar', 'BodyTemp', 'HeartRate', 'RiskLevel']

    #TUPLE RIDONDANTI 562
    print(dataset.duplicated())
    #dataset = dataset.drop_duplicates()

    categoricalColumn = [col for col in dataset.columns if (dataset[col].dtype == 'object' )]
    numericalColumn = [col for col in dataset.columns if col not in categoricalColumn]

    print('\nCategorical columns:')
    print(categoricalColumn)
    print('\nNumerical columns:')
    print(numericalColumn)
    print('\n\n')

    #PRESENZA DI VALORI ANOMALI, CONTROLLIAMO
    outlierDetection(dataset['Age'])
    outlierDetection(dataset['SystolicBloodPressure'])
    outlierDetection(dataset['DiastolicBloodPressure'])
    outlierDetection(dataset['BloodSugar'])
    outlierDetection(dataset['BodyTemp'])
    outlierDetection(dataset['HeartRate'])

    dataset['RiskLevel'] = dataset['RiskLevel'].map({'low risk':0,'mid risk':1,'high risk':2})
    
    categPlotShow(dataset)
    numPlotShow(dataset,"Age")
    numPlotShow(dataset,"SystolicBloodPressure")
    numPlotShow(dataset,"DiastolicBloodPressure")
    numPlotShow(dataset,"BloodSugar")
    numPlotShow(dataset,"BodyTemp")
    numPlotShow(dataset,"HeartRate")

    scatterPlotShow(dataset,"Age","DiastolicBloodPressure")
    scatterPlotShow(dataset,"BloodSugar","BodyTemp")
    scatterPlotShow(dataset,"SystolicBloodPressure","BodyTemp")
    scatterPlotShow(dataset,"Age","HeartRate")
    scatterPlotShow(dataset,"SystolicBloodPressure","HeartRate")
    scatterPlotShow(dataset,"DiastolicBloodPressure","HeartRate")
    scatterPlotShow(dataset,"BloodSugar","HeartRate")
    scatterPlotShow(dataset,"BodyTemp","HeartRate")
    numericData = dataset.drop("RiskLevel", axis=1)
    corrHeatMapShow(numericData)
    
    #Dallo zscore sembrano esserci dei valori anomali, ma in questo contesto potrebbero essere delle reali registrazioni.
    #DATA LA PRESENZA DI VALORI ANOMALI USIAMO IL CLUSTERING PER VERIFICARE MEGLIO LA PRESENZA DI ALTRI VALORI ANOMALI IMPORTANTI
    target = dataset['RiskLevel']
    data = dataset.drop("RiskLevel", axis=1)
    data = dataset
    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(data)
    dataReduced = reduceData(dataScaled,3)
    metodKMeans(dataReduced,target)

    printInfo(dataset)
    control(dataset)

