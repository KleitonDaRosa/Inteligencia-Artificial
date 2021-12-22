#Algoritmo de Reqressão Linear Simples
#__Kleiton Da Rosa Delgado
#Seminario Inteligencia Artificial 22 Dezembro 2021



#importando as bibliotecas necessarias

import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#Importanto a DataSet
dados_salario = pd.read_csv("https://raw.githubusercontent.com/ect-info/ml/master/dados/Salary_Data.csv")
#Visualizando as primeiras linhas da tabela da base de dados
#print(dados_salario.head())

#Separaçao dos dados para variavel independende e para a variavel dependente
Xv = dados_salario.iloc[:,0].values
y = dados_salario.iloc[:,1].values

# Converter X, vetor n elementos, para uma matriz de n x 1
Xm = np.array([Xv])
X = Xm.T

print(X)

# Cria um objeto para a regressão linear
regre = linear_model.LinearRegression()


# Realiza o ajuste do modelo
regre.fit(X,y)

# Obtem os valores ajustados
y_est = regre.predict(X)


plt.scatter(X,y,color='red')
plt.plot(X,y_est, color='blue', linewidth=2)
plt.show()



