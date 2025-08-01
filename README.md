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

Praticamente todos os parâmetros **não apresentam** grande correlação com o total de casos, portanto, foi necessário estabelecer um **valor mínimo** para que os parâmetros fossem filtrados. Tal limite foi de **10 %**. E assim, os seguintes parâmetros foram escolhidos, por cidade:

### ``San Juan``
- ``reanalysis_air_temp_k``
- ``reanalysis_avg_temp_k``
- ``reanalysis_dew_point_temp_k``
- ``reanalysis_max_air_temp_k``
- ``reanalysis_min_air_temp_k``
- ``reanalysis_precip_amt_kg_per_m2``
- ``reanalysis_relative_humidity_percent``
- ``reanalysis_specific_humidity_g_per_kg``
- ``station_avg_temp_c``
- ``station_max_temp_c``
- ``station_min_temp_c``

## ``ii.`` Treinamento da rede neural

Para tentar predizer o total de casos, utilizei o modelo ``Sequential`` da biblioteca ``Keras`` adicionando três camadas densas com 16, 8 e 1 neurônios, respectivamente. Outras arquiteturas foram testadas, como por exemplo (64,32,1), (34,32,16,1), porém, a arquitetura mencionada anteriormente apresentou os melhores resultados.

```python
model = Sequential()

model.add(Dense(units = 16, activation = keras.layers.LeakyReLU()))
model.add(Dense(units = 8, activation = keras.layers.LeakyReLU()))
model.add(Dense(units = 1))

model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.MeanSquaredError(),
              metrics = [keras.losses.MeanAbsoluteError()])
```

A função de ativação utilizada foi a **LeakyReLU** para evitar a morte de neurônios.

