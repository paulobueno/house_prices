import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from scipy import stats
from sklearn import linear_model

data_1 = pd.read_csv('./train.csv')

#remove colunas com missing maior que 50%
#preenche colunas numericas por mediana e categoricas por moda

tabela = pd.DataFrame(data_1)

def p_missing(x):
    if x.isnull().sum()/len(x) < 0.5:
        return True
    else:
        return False

def f_missing(x):
    if(x.dtype == np.float64 or x.dtype == np.int64):
        return x.fillna(x.median(), inplace=True)
    else:
        return x.fillna(x.mode()[0], inplace=True)
    
var_list = (tabela.apply(p_missing, axis=0))
tabela = tabela.loc[:, var_list]

for x in tabela.columns:
    f_missing(tabela[x])

#Remove colunas não numéricas
tabela = tabela.loc[:, tabela.dtypes != object]

#Cria dataset para treino, desconsiderando as 20 últimas linhas
tabela_treino_x = tabela.drop('SalePrice',1)[:-20]

#Cria dataset para teste, considerando apenas as últimas 20 linhas
tabela_validacao_x = tabela.drop('SalePrice',1)[-20:]

# Cria datasets com a variável resposta
tabela_treino_y = tabela.SalePrice[:-20]
tabela_validacao_y = tabela.SalePrice[-20:]

print('Coefficients: \n', regr.coef_)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(tabela_validacao_x, tabela_validacao_y))
