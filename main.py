import seaborn as sns


# Função para dividir o conjunto de dados de treinamento em janelas
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

# Importa os parâmetros para ajuste (labels) e a variável a ser predita (feature)
labels = pd.read_csv("diretório de armazenamento/dengue_features_train.csv")
feature = pd.read_csv("diretório de armazenamento/dengue_labels_train.csv")

# Separa os labels e features por cidade
labels_sj = labels[labels["city"] == "sj"]
feature_sj = feature[feature["city"] == "sj"]

labels_iq = labels[labels["city"] == "iq"]
feature_iq = feature[feature["city"] == "iq"]

# Junta labels e features para facilitar o tratamento
labels_sj_join = labels_sj.join(feature_sj['total_cases'])
labels_iq_join = labels_iq.join(feature_iq['total_cases'])

# Retira colunas que não fazem parte da análise (e escolhe qual cidade analisar)
notUsedColumns = ['year', 'weekofyear', 'week_start_date', 'city']

# San Juan
labels_join = labels_sj_join.drop(notUsedColumns, axis = 1).copy()

# Iquitos
# labels_join = labels_iq_join.drop(notUsedColumns, axis = 1).copy()

# Plota a matriz de correlação geral

# Calcula correlação com todos os parâmetros
corrMatrix = labels_join.corr()

sns.heatmap(corrMatrix, center = 0)

# Filtra os parâmetro com correlação acima de um limite

# Estabelece um limite para correlação
threshold = 0.10

# Lista que receberá as colunas com correlação acima do treshold
columnNames = []

# Encontra os parâmetros com correlação acima do treshold
for col in corrMatrix.columns:
  if np.abs(corrMatrix.loc['total_cases', col]) > threshold:
    columnNames.append(col)

# DataFrame que receberá a matrix de correlação dos parâmetros significativos
corrMatrixSignificant = pd.DataFrame()

for row in columnNames:
  for col in columnNames:
    corrMatrixSignificant.loc[row, col] = corrMatrix.loc[row, col]

sns.heatmap(corrMatrixSignificant, center = 0)

# Pega somente os dados dos parâmetros com correlação significativa
labels_join_training = labels_join[columnNames]
labels_join_training = labels_join_training.dropna()

# Separa labels e features
labels_training = labels_join_training.drop(['total_cases'], axis = 1).copy()
feature_training = labels_join_training['total_cases'].copy()

# Transforma o DataFrame em array
labelsArray_training = labels_training.to_numpy()
featureArray_training = feature_training.to_numpy()

# Reescala o conjunto de dados para ter média nula e desvio padrão unitário
labelsArrayNorm_training = (labelsArray_training - np.mean(labelsArray_training, axis = 0))/np.std(labelsArray_training, axis = 0)
featureArrayNorm_training = (featureArray_training - np.mean(featureArray_training, axis = 0))/np.std(featureArray_training, axis = 0)

# Separa o conjunto de treinamento e validação
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

# Prediz para o conjunto de treinamento e validação
predNormTrain = model.predict(xTrainNorm)
predNormVal = model.predict(xValNorm)

# Retorna os dados à escala original
predTrain = predNormTrain*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)
predVal = predNormVal*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)
yTrain = yTrainNorm*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)
yVal = yValNorm*np.std(featureArray_training, axis = 0) + np.mean(featureArray_training, axis = 0)

# Verifica o MAE para dados de treino e validação
from sklearn.metrics import mean_absolute_error

print(f"MAE - Treinamento: {mean_absolute_error(predTrain, yTrain)}")
print(f"MAE - Validação: {mean_absolute_error(predVal, yVal)}")

#################
# PARA SAN JUAN #
#################

# Calcula a média móvel
import pandas as pd

dfVal = pd.DataFrame({'Predictions': [i[0] for i in predVal]})
dfVal['rolling average'] = dfVal.rolling(7, min_periods = 1).mean()
rollAvgVal = dfVal['rolling average'].to_numpy()

# Verifica o MAE
print(f"MAE: {mean_absolute_error(rollAvgVal, yVal)}")

# Ajuste das predições com base na diferença entre os valores da média móvel, e os dados reais
# Calcula a diferença entre as predições suavizadas e os dados de validação
dif = rollAvgVal - yVal
# Toma a média da diferença pois será utilizada para rebaixar os dados de teste
meanDif = np.mean(dif)
# Rebaixa os dados da predição pela média da diferença
rollAvgVal_mod = rollAvgVal - meanDif

# Verifica o MAE final
print(f"MAE: {mean_absolute_error(rollAvgVal_mod, yVal)}")