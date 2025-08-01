# ``Projeto DengAI``
Predição de casos de dengue nas cidades San Juan, Porto Rico, e Iquitos, Peru, através de um modelo de Deep Neural Network.

## ``i.`` Parâmetros para treinamento
Os parâmetros para treinamento são basicamente dados climáticos, como um índice de vegetação das cidades, dados de temperatura, umidade e precipitação. A aquisição dos dados ocorreu de forma semanal, e todos compõe séries temporais ao longo de vários anos. Neste <a href="https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/82/">link</a> é possível ver a descrição de todos os parâmetros.

Como exemplo, abaixo vemos os dados do **total de casos** de dengue, e da **temperatura máxima**.

<img width="1431" height="514" alt="Image" src="https://github.com/user-attachments/assets/efbb4333-7788-461d-b27f-4558d371949a" />

Ao invés de verificar visualmente quais parâmetros se correlacionam com a variável que queremos predizer, o total de casos, vejamos suas **correlações de Pearson** por meio de uma matriz de correlação.

