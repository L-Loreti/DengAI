import seaborn as sns


# Fun��o para dividir o conjunto de dados de treinamento em janelas
def windowing(labels, features, window_size):

  i = 0
  listFeatures = []
  listLabels = []

  while i <= len(features) - window_size:

    for j in range(window_size):
      listFeatures.append(features[i + j])
      listLabels.append(labels[i + j])

    i += 1

  return np.array(listLabels), np.array(listFeatures)

# Importa os par�metros para ajuste (labels) e a vari�vel a ser predita (feature)
labels = pd.read_csv("diret�rio de armazenamento/dengue_features_train.csv")
feature = pd.read_csv("diret�rio de armazenamento/dengue_labels_train.csv")

# Separa os labels e features por cidade
labels_sj = labels[labels["city"] == "sj"]
feature_sj = feature[feature["city"] == "sj"]

labels_iq = labels[labels["city"] == "iq"]
feature_iq = feature[feature["city"] == "iq"]

# Junta labels e features para facilitar o tratamento
labels_sj_join = labels_sj.join(feature_sj['total_cases'])
labels_iq_join = labels_iq.join(feature_iq['total_cases'])

# Retira colunas que n�o fazem parte da an�lise (e escolhe qual cidade analisar)
notUsedColumns = ['year', 'weekofyear', 'week_start_date', 'city']

# San Juan
labels_join = labels_sj_join.drop(notUsedColumns, axis = 1).copy()

# Iquitos
# labels_join = labels_iq_join.drop(notUsedColumns, axis = 1).copy()

# Plota a matriz de correla��o geral

# Calcula correla��o com todos os par�metros
corrMatrix = labels_join.corr()

sns.heatmap(corrMatrix, center = 0)

# Filtra os par�metro com correla��o acima de um limite

# Estabelece um limite para correla��o
threshold = 0.10

# Lista que receber� as colunas com correla��o acima do treshold
columnNames = []

# Encontra os par�metros com correla��o acima do treshold
for col in corrMatrix.columns:
  if np.abs(corrMatrix.loc['total_cases', col]) > threshold:
    columnNames.append(col)

# DataFrame que receber� a matrix de correla��o dos par�metros significativos
corrMatrixSignificant = pd.DataFrame()

for row in columnNames:
  for col in columnNames:
    corrMatrixSignificant.loc[row, col] = corrMatrix.loc[row, col]

sns.heatmap(corrMatrixSignificant, center = 0)

# Pega somente os dados dos par�metros com correla��o significativa
labels_join_training = labels_join[columnNames]
labels_join_training = labels_join_training.dropna()

# Separa labels e features
labels_training = labels_join_training.drop(['total_cases'], axis = 1).copy()
feature_training = labels_join_training['total_cases'].copy()

# Transforma o DataFrame em array
labelsArray_training = labels_training.to_numpy()
featureArray_training = feature_training.to_numpy()

# Reescala o conjunto de dados para ter m�dia nula e desvio padr�o unit�rio
labelsArrayNorm_training = (labelsArray_training - np.mean(labelsArray_training, axis = 0))/np.std(labelsArray_training, axis = 0)
featureArrayNorm_training = (featureArray_training - np.mean(featureArray_training, axis = 0))/np.std(featureArray_training, axis = 0)

# Separa o conjunto de treinamento e valida��o
from sklearn.model_selection import train_test_split

xTrainNorm, xValNorm, yTrainNorm, yValNorm = train_test_split(labelsArrayNorm_training, featureArrayNorm_training, test_size = 0.30, shuffle = False)

# Prepara o conjunto de treinamento em janelas
window_size = 1
xTrainNorm_windowed, yTrainNorm_windowed = windowing(xTrainNorm, yTrainNorm, window_size)

# Declara a rede neural
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

model = Sequential()

model.add(Dense(units = 16, activation = keras.layers.LeakyReLU()))
model.add(Dense(units = 8, activation = keras.layers.LeakyReLU()))
model.add(Dense(units = 1))

model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.MeanSquaredError(),
              metrics = [keras.losses.MeanAbsoluteError()])

model.summary()

# Ajusta os pesos da rede neural
epochs = 20
history = model.fit(xTrainNorm_windowed, yTrainNorm_windowed, epochs = epochs, batch_size = window_size)

# Prediz para o conjunto de treinamento e valida��o
predNormTrain = model.predict(xTrainNorm)
predNormVal = model.predict(xValNorm)

# Retorna os dados � escala original
predTrain = predNormTrain*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)
predVal = predNormVal*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)
yTrain = yTrainNorm*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)
yVal = yValNorm*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)

# Verifica o MAE para dados de treino e valida��o
from sklearn.metrics import mean_absolute_error

print(f"MAE - Treinamento: {mean_absolute_error(predTrain, yTrain)}")
print(f"MAE - Valida��o: {mean_absolute_error(predVal, yVal)}")

#################
# PARA SAN JUAN #
#################

# Calcula a m�dia m�vel
import pandas as pd

dfVal = pd.DataFrame({'Predictions': [i[0] for i in predVal]})
dfVal['rolling average'] = dfVal.rolling(7, min_periods = 1).mean()
rollAvgVal = dfVal['rolling average'].to_numpy()

# Verifica o MAE
print(f"MAE: {mean_absolute_error(rollAvgVal, yVal)}")

# Ajuste das predi��es com base na diferen�a entre os valores da m�dia m�vel, e os dados reais
# Calcula a diferen�a entre as predi��es suavizadas e os dados de valida��o
dif = rollAvgVal - yVal
# Toma a m�dia da diferen�a pois ser� utilizada para rebaixar os dados de teste
meanDif = np.mean(dif)
# Rebaixa os dados da predi��o pela m�dia da diferen�a
rollAvgVal_mod = rollAvgVal - meanDif

# Verifica o MAE final
print(f"MAE: {mean_absolute_error(rollAvgVal_mod, yVal)}")