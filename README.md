# ``Projeto DengAI``
Predição de casos de dengue nas cidades San Juan, Porto Rico, e Iquitos, Peru, através de um modelo de Deep Neural Network.

## ``i.`` Parâmetros para treinamento
Os parâmetros para treinamento são basicamente dados climáticos, como um índice de vegetação das cidades, dados de temperatura, umidade e precipitação. A aquisição dos dados ocorreu de forma semanal, e todos compõe séries temporais ao longo de vários anos. Neste <a href="https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/82/">link</a> é possível ver a descrição de todos os parâmetros.

Como exemplo, abaixo vemos os dados do **total de casos** de dengue, e da **temperatura máxima**, fora de suas escalas originais.

<img width="1431" height="514" alt="Image" src="https://github.com/user-attachments/assets/efbb4333-7788-461d-b27f-4558d371949a" />

Ao invés de verificar visualmente quais parâmetros se correlacionam com a variável que queremos predizer, o total de casos, vejamos suas **correlações de Pearson** por meio de uma matriz de correlação.

<p align = "center">
<img width="700" height="600" alt="Image" src="https://github.com/user-attachments/assets/f82a9e66-ba9b-4abd-a960-2361684e419f" />
</p>

Praticamente todos os parâmetros **não apresentam** grande correlação com o total de casos, portanto, foi necessário estabelecer um **valor mínimo** para que os parâmetros fossem filtrados. Tal limite foi de **10 %** para os dados das duas cidades, pois caso fosse maior, poucos parâmetros comporiam o modelo. Abaixo relaciono os parâmetros escolhidos, por cidade:

### ``San Juan (11 parâmetros)``
- _reanalysis_air_temp_k_
- _reanalysis_avg_temp_k_
- _reanalysis_dew_point_temp_k_
- _reanalysis_max_air_temp_k_
- _reanalysis_min_air_temp_k_
- _reanalysis_precip_amt_kg_per_m2_
- _reanalysis_relative_humidity_percent_
- _reanalysis_specific_humidity_g_per_kg_
- _station_avg_temp_c_
- _station_max_temp_c_
- _station_min_temp_c_

### ``Iquitos (8 parâmetros)``

- _reanalysis_dew_point_temp_k_
- _reanalysis_min_air_temp_k_
- _reanalysis_precip_amt_kg_per_m2_
- _reanalysis_relative_humidity_percent_
- _reanalysis_specific_humidity_g_per_kg_
- _reanalysis_tdtr_k_
- _station_avg_temp_c_
- _station_min_temp_c_

## ``ii.`` Treinamento da rede neural

Para tentar predizer o total de casos, utilizei o modelo ``Sequential`` da biblioteca ``Keras`` adicionando três camadas densas com _16_, _8_ e _1_ neurônios, respectivamente. Outras arquiteturas foram testadas, como por exemplo _(64,32,1)_, _(64,32,16,1)_, porém, a arquitetura mencionada anteriormente apresentou os melhores resultados.

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units = 16, activation = keras.layers.LeakyReLU()))
model.add(Dense(units = 8, activation = keras.layers.LeakyReLU()))
model.add(Dense(units = 1))

model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.MeanSquaredError(),
              metrics = [keras.losses.MeanAbsoluteError()])
```

A função de ativação utilizada foi a **LeakyReLU** para evitar a morte de neurônios. 

O conjunto de dados disponibilizado pelo <a href="https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/">site<a/> foi dividido em 70% para treinamento e 30% para validação.

```Python
from sklearn.model_selection import train_test_split

xTrainNorm, xValNorm, yTrainNorm, yValNorm = train_test_split(labelsArrayNorm_training, featureArrayNorm_training,
                                                              test_size = 0.30, shuffle = False)
```

Onde ``labelsArrayNorm_training`` e ``featureArrayNorm_training`` são uma matriz contendo os parâmetros escolhidos para treinamento, e a variável que queremos predizer, respectivamente. Ambas foram normalizadas de modo a terem **média nula** e **desvio padrão unitário**, para que as importâncias dos parâmetros fossem equivalentes.

Além disso, testei diferentes **janelas** de treinamento para os dados, de modo a tentar capturar comportamentos de longo prazo, no entanto, os melhores resultados foram obtidos com **window_size = 1**. Por fim, o número de **épocas** de treinamento foi escolhido como 20, pois além disso, o modelo não melhorava a _erro absoluto médio_.

```python
epochs = 20
window_size = 1
history = model.fit(xTrainNorm, yTrainNorm, epochs = epochs, batch_size = window_size)
```

## ``iii.`` Resultados

A métrica utilizada para avaliar o ajuste do modelo aos dados de treinamento e validação foi o _Erro Absoluto Médio_ (_Mean Absolute Error - MAE_), o qual está relacionado abaixo, juntamente com duas imagens do ajuste para a cidade de **San Juan**.

- ``San Juan - Conjunto de treinamento``
<img width="8865" height="3289" alt="Image" src="https://github.com/user-attachments/assets/0478ba04-3e35-4999-83f2-f05e243cfdce" />

**MAE - Treinamento**: 28.45

- ``San Juan - Conjunto de validação``
<img width="8785" height="3281" alt="Image" src="https://github.com/user-attachments/assets/cd5bbf17-6c08-42ab-ac80-54882c7821a2" />

**MAE - Validação**: 26.19

Em todos os testes realizados para a cidade **San Juan**, o modelo retornou predições superestimadas, enquanto que para **Iquitos** as predições estavam com um MAE pequeno, como por ser visto no gráfico abaixo.

- ``Iquitos - Conjunto de treinamento``

<img width="8761" height="3273" alt="Image" src="https://github.com/user-attachments/assets/a7c40c99-2c4c-4ad5-93cf-9ddde9552143" />

**MAE - Treinamento**: 4.65

- ``Iquitos - Conjunto de validação``

<img width="8729" height="3257" alt="Image" src="https://github.com/user-attachments/assets/6d871457-503c-4dc4-a7d8-5eabeddca7bb" />

**MAE - Validação**: 7.87

Para melhorar as predições para **San Juan**, tentei abaixá-las utilizando médias móveis, como descrito no próximo tópico.

## ``iv.`` Ajuste dos resultados com médias móveis

Inicialmente, tomei a média móvel de um período de sete semanas para os valores preditos pelo modelo. Utilizei a função ``rolling()`` dos ``DataFrames`` da biblioteca ``Pandas``. É importante notar o uso do argumento ``min_periods = 1``, pois assim a média móvel é calculada para intervalos de dados menores que sete semanas, não reduzindo a quantidade total de predições.

```Python
import pandas as pd

dfVal = pd.DataFrame({'Predictions': [i[0] for i in predVal]})
dfVal['rolling average'] = dfVal.rolling(7, min_periods = 1).mean()
rollAvgVal = dfVal['rolling average'].to_numpy()
```
Onde ``predVal`` é um ``array`` dos valores preditos pelo modelo, já **desnormalizados**.

Da figura abaixo, vemos a média móvel junto com as predições.

<img width="8825" height="3289" alt="Image" src="https://github.com/user-attachments/assets/16dca176-a297-44c4-9ca8-98f22640af7a" />

Já nesse caso, somente com a suavização da curva devido à média móvel, a métrica *MAE* é reduzida para 24.93.

O intuito agora é calcular a diferença média entre os valores da média móvel, e os dados reais, e subtraí-la dos valores da média móvel rebaixando-a.

```Python
# Calcula a diferença entre as predições suavizadas e os dados de validação
dif = rollAvgVal - yVal
# Toma a média da diferença pois será utilizada para rebaixar os dados de teste
meanDif = np.mean(dif)
# Rebaixa os dados da predição pela média da diferença, sendo rllAvgVal, os valores da média móvel para as predições
rollAvgVal_mod = rollAvgVal - meanDif
```

Do gráfico abaixo vemos o resultado, com **MAE**: 17.62.

<img width="8761" height="3257" alt="Image" src="https://github.com/user-attachments/assets/0215b92d-6840-4072-9681-5c484d647c3d" />
