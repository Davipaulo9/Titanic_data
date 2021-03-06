#!/usr/bin/env python
# coding: utf-8

# ## Project 4 - UDACITY_NANODEGREE_FUNDAMENTALS_OF DATA_SCIENCE
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# #Quais foram os fatores que fizeram com que algumas pessoas fossem mais propensas a sobreviver?
# 
# References:
# https://www.kaggle.com/c/titanic
# http://www.chasingthefrog.com/reelfaces/titanic.php

# In[405]:


#Load libraries
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')


# #Importing Dataset

# In[406]:


df = pd.read_csv('train.csv')


# #Checking rows and columns

# In[407]:


df.shape


# #Lets visualize  the first 5 entries of data

# In[408]:


df.head()


# #1 Quantos passageiros havia em cada classe do navio?
# #2 Quantos adultos haviam no navio?
# #3 Quantas crianças haviam a bordo do navio?
# #4 Quantas mulheres e homens haviam a bordo?
# #5 Quantas vidas foram salvas?
# #6 Quantas vidas foram perdidas?
# #7 Quantos passageiros embarcaram em cada porto?
# #8 Quantos passageiros embarcaram em cada classe em cada porto?
# #9 Qual a média das idades das mulheres?
# #10 Qual a média das idades dos homens?
# #11 Qual a idade da mulher mais idosa?
# #12 Qual a idade do homem mais idoso?12
# #13 Quantos homens sobreviveram?
# #14 Quantas mulheres sobreviveram? 
# #15 Quantos eram garotos?
# #16 Quantos eram garotas?
# #17 Qual era a porcentagem de crianças?
# #18 Qual era a procentagem de mulheres?
# #19 Qual era a procentagem de homens?
# #20 Quantas crianças foram salvas?
# #21 Quantas garotos foram salvas?
# #22 Quantas garotas foram salvas?
#  
# # Let's find the survivors by classes
# 
# #23 Quantas mulheres da primeira classe sobreviveram?
# #24 Quantas mulheres da segunda classe sobreviveram?
# #25 Quantas mulheres da terceira classe sobreviveram?
# #26 Quantos homens da primeira classe sobreviveram?
# #27 Quantos homens da segunda classe sobreviveram?
# #28 Quantos homens da terceira classe sobreviveram?
# 
# 
# #29 Quantas garotas da primeira classe sobreviveram?
# #30 Quantas garotas da segunda classe sobreviveram?
# #31 Quantas garotas da terceira classe sobreviveram?
# #32 Quantas garotos da terceira classe sobreviveram?
# #33 Quantas garotos da terceira classe sobreviveram?
# #34 Quantas garotos da terceira classe sobreviveram?
# 
# #35 Vamos visualizar os sobreviventes por idade
# #36 Visulizar ticket médio por Classe
# 
# # Adults
# #38 How many Adults Females survived from the 1st Class that board in Queenstown?
# #39 How many Adults Females survived from the 1st Class that board in Cherborough? 
# #40 How many Adults Females survived from the 1st Class that board in Southhampton?
# #41 How many Adults Females survived from the 2nd Class that board in Queenstown?
# #42 How many Adults Females that survived from the 2nd Class board in Cherborough?
# #43 How many Adults Females that survived from the 1st Class board in Queenstown?
# #44 How many Adults Females that survived from the 3rd Class board in Queenstown?
# #45 How many Adults Females that survived from the 3rd Class board in Cherborough?
# #46 How many Adults Females that survived from the 3rd Class board in Queenstown?
# #47 How many Adults Males that survived from the 1st Class board in Queenstown?
# #48 How many Adults Males that survived from the 1st Class board in Cherborough?
# #49 How many Adults Males that survived from the 1st Class board in Southhampton?
# #50 How many Adults Males that survived from the 2nd Class board in Queenstown?
# #51 How many Adults Males that survived from the 2nd Class board in Queenstown? 
# #52 How many Adults Males that survived from the 1st Class board in Queenstown? 
# #53 How many Adults Males that survived from the 3rd Class board in Queenstown?
# #54 How many Adults Males that survived from the 3rd Class board in Cherborough?
# #55 How many Adults Males that survived from the 3rd Class board in Queenstown?
# 
# 
# # Children
# #56 How many girls that survived from the 1st Class board in Queenstown? 
# #57 How many girls that survived from the 2nd Class board in Cherborough?
# #58 How many girls that survived from the 1st Class that in Queenstown? 
# #59 How many girls that survived from the 2nd Class board in Queenstown?
# #60 How many girls that survived from the 2nd Class board in Cherborough?
# #61 How many girls that survived from the 2nd Class board in Southhampton?
# #62 How many girls that survived from the 2nd Class that board in Queenstown? 
# #63 How many girls that survived from the 2nd Class board in Cherborough?
# #64 How many girls that survived from the 2nd Class board in Queenstown?
# #65 How many boys that survived from the 1st Class board in Queenstown?
# #66 How many boys that survived from the 1st Class that in Cherborough?
# #67 How many boys that survived from the 1st Class board in Queenstown?
# #68 How many boys that survived from the 2nd Class board in Queenstown? 
# #69 How many boys that survived from the 2nd Class that in Cherborough?
# #70 How many boys that survived from the 2nd Class board in Southhampton?
# #71 How many boys that survived from the 2nd Class board in Queenstown? 
# #72 How many boys that survived from the 2nd Class that in Cherborough?
# #73 How many boys that survived from the 2nd Class board in Queenstown?
# 
# #74 Qual a idade média dos sobreviventes do naufrágio?
# #75 Qual a idade média das vítimas do naufrágio?
# #76 Qual a idade do sobrevivente mais novo?
# #77 Qual a idade da vítima mais nova?
# #78 Qual a idade do sobrevivente mais idoso?
# #79 Qual a idade da vítima mais idosa?
# #77 Qual a idade do sobrevivente mais idoso?
# #78 Qual a idade da vítima mais idosa?
# 
# Qual a porcentagem de sobreviventes em cada classe do navio?
# Qual a porcentagem de sobreviventes entre as crianças?
# Qual a porcentagem de sobreviventes entre as mulheres?
# Qual a porcentagem de sobreviventes entre os homens?
# Qual a porcentagem de sobreviventes entre os idosos?
# Qual a porcentagem de sobreviventes entre os que viajavam com a família?
# Qual a porcentagem de sobreviventes entre os que viajavam desacompanhados?
# Qual o valor médio dos tíquetes de sobreviventes em cada classe do navio?
# Qual o valor médio dos tíquetes das vítimas em cada classe do navio?
# Quais fatores contribuíram para a sobrevivência dos passageiros a bordo?

# #Let's clean the data since we have lots of missing information

# # Limpeza de dados

# #Converter colunas Float para Int
# #Preencher dados vazios
# #Remover valores vazios da coluna Embarked

# In[409]:


df.head()


# In[410]:


df['Embarked'].value_counts().sum()
df.dropna(subset=['Embarked'], inplace=True) 
df.shape


# #Converter colunas Float para Int

# In[411]:


i = df.loc[df.Age > 2, 'Age'].mean()


# #Converter colunas Float para Int

# In[412]:


df['Fare'] = df['Fare'].astype(int)
df.head()


# #Remover valores vazios da coluna Embarked

# In[413]:


df['Embarked'].value_counts().sum()
df.dropna(subset=['Embarked'], inplace=True) 
df.shape


# In[ ]:





# # Fim da limpeza dos dados

# #1 Quantos passageiros havia em cada classe do navio?

# In[414]:


classes = df.groupby('Pclass').size()
print(classes)
print('There were 184 passengers at Class 1, 172 at Class 2 and 255 at Class 3')


# #2 Quantos adultos haviam no navio?

# In[415]:


adults  = df[(df.Age > 18)].count().astype(int)
adults = adults['Age']
adults = 'There were {}  adults'.format(adults)
print(adults)


# #3 Quantas crianças haviam a bordo?

# In[416]:


children = df[(df.Age <= 18)].count().astype(int)
children = children['Age']
children = 'There were {}  children'.format(children)
print(children)


#  #4 Quantas mulheres e homens haviam a bordo

# In[417]:


counts = df['Sex'].value_counts()
print(counts)
print('n\There were 453 Men and 259 women aboard, including children')


# #5 Quantas vidas foram salvas?

# In[418]:


survivors = df[(df.Survived == 1)].count()
survivors = survivors['Survived']
survivors = 'There were {} survivors'.format(survivors)
print(survivors) 


# #6 Quantas vidas foram perdidas?

# In[419]:


df_lost = df.loc[df['Survived'] == 0].count().astype(int)
df_lost = df_lost['Survived']
df_lost = 'There were {} lives lost'.format(df_lost) 
print (df_lost)


# #7 Quantos embarcaram em cada porto?

# In[420]:


df['Embarked'].value_counts()


# #8 Quantos passageiros embarcaram em cada classe em cada porto?

# 1ST CLASS

# In[421]:


class_1_south = df[(df.Pclass == 1) & (df.Embarked == 'S')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 1st Class who embarked in Southhampton'.format(class_1_south)
print(class_1_south)


# In[422]:


class_1_south = df[(df.Pclass == 1) & (df.Embarked == 'C')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 1st Class who embarked in Cherbourgh'.format(class_1_south)
print(class_1_south)


# In[423]:


class_1_south = df[(df.Pclass == 1) & (df.Embarked == 'Q')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 1st Class who embarked in Queenstown'.format(class_1_south)
print(class_1_south)


# 2ND CLASS

# In[424]:


class_1_south = df[(df.Pclass == 2) & (df.Embarked == 'S')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 2st Class who embarked in Southhampton'.format(class_1_south)
print(class_1_south)


# In[425]:


class_1_south = df[(df.Pclass == 2) & (df.Embarked == 'C')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 2st Class who embarked in Cherbourgh'.format(class_1_south)
print(class_1_south)


# In[426]:


class_1_south = df[(df.Pclass == 2) & (df.Embarked == 'Q')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 2st Class who embarked in Queenstown'.format(class_1_south)
print(class_1_south)


# 3RD CLASS

# In[427]:


class_1_south = df[(df.Pclass == 2) & (df.Embarked == 'S')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 2st Class who embarked in Southhampton'.format(class_1_south)
print(class_1_south)


# In[428]:


class_1_south = df[(df.Pclass == 2) & (df.Embarked == 'C')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 2st Class who embarked in Cherbourgh'.format(class_1_south)
print(class_1_south)


# In[429]:


class_1_south = df[(df.Pclass == 2) & (df.Embarked == 'Q')].count().astype(int)
class_1_south = class_1_south['Pclass']
class_1_south = 'There were {} passengers from 2st Class who embarked in Queenstown'.format(class_1_south)
print(class_1_south)


# #9 Qual a média das idades das mulheres?

# In[430]:


df1 = df[df['Sex'] == 'female'] 
df2 = df[df['Sex'] == 'male']
df1['Age'].mean() 


# #10 Qual a média das idades dos homens?

# In[431]:


df1 = df[df['Sex'] == 'female'] 
df2 = df[df['Sex'] == 'male']
df2['Age'].mean()


# #11 Qual a idade da mulher mais idosa?

# In[432]:


df1['Age'].max()


# #12 Qual a idade do homem mais idoso?

# In[433]:


df2['Age'].max()


# #13 Quantos homens sobreviveram?

# In[434]:


df_saved  = df[(df.Sex == 'male') & (df.Age > 18) &(df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = '{} male adults survived'.format(df_saved)
print(df_saved)


# #14 Quantas mulheres sobreviveram? 

# In[435]:


df_saved  = df[(df.Sex == 'female') & (df.Age > 18) &(df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} female adults survived'.format(df_saved)
print(df_saved)


# #15 Quantos eram garotos?

# In[436]:


boys = df[(df.Sex == 'male') & (df.Age <= 18)].count().astype(int)
boys = boys['Sex']
boys = 'There were {} boys'.format(boys)
print(boys)


# #16 Quantos eram garotas?

# In[437]:


girls = df[(df.Sex == 'female') & (df.Age <= 18)].count().astype(int)
girls = girls['Sex']
girls = 'There were {} girls'.format(girls)
print(girls)


# #17 Qual era a porcentagem de crianças

# In[438]:


139 / 712 * 100


# #18 Qual era a porcentagem de mulheres?

# In[439]:


259 / 712 * 100


# #19 Qual era a procentagem de homens?

# In[440]:


453 / 712 * 100


# #20 Quantas crianças foram salvas?

# In[441]:


df_saved = df[(df.Age <= 18) & (df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} children saved'.format(df_saved)
print(df_saved)


# #21 Quantas garotos foram salvos?

# In[442]:


df_saved = df[(df.Sex == 'male') & (df.Age <= 18) & (df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} boys saved'.format(df_saved)
print(df_saved)


# #22 Quantas garotas foram salvas?

# In[443]:


df_saved = female_count = df[(df.Sex == 'female') & (df.Age <= 18) & (df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} girls saved'.format(df_saved)
print(df_saved) 


# # Let's find the survivors by classes

# #23 Quantas mulheres da primeira classe sobreviveram?

# In[444]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1)]
first_class = ffc_sur
first_class = ffc_sur[first_class.columns[1:5]]
first_class.head(5)
qty = (len(first_class))
qty = 'There are {} female survivors from the First Class'.format(qty)
print(qty)     


# #24 Quantas mulheres da segunda classe sobreviveram?

# In[445]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2)]
first_class = ffc_sur
first_class = ffc_sur[first_class.columns[1:5]]
first_class.head(5)
qty = (len(first_class))
qty = 'There are {} female survivors from the Second Class'.format(qty)
print(qty)  


# #25 Quantas mulheres da terceira classe sobreviveram?

# In[446]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 3) & (df.Pclass == 1)]
first_class = ffc_sur
first_class = ffc_sur[first_class.columns[1:5]]
first_class.head(5)
qty = (len(first_class))
qty = 'There are {} female survivors from the Third Class'.format(qty)
print(qty)  


# #26 Quantos homens da primeira classe sobreviveram?

# In[447]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1)]
first_class = ffc_sur[first_class.columns[1:4]]
first_class.head(5) 
qty = (len(first_class))
qty = 'There are {} male survivors from the first Class'.format(qty)
print(qty)      


# #27 Quantos homens da segunda classe sobreviveram?

# In[448]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2)]
first_class = ffc_sur[first_class.columns[1:4]]
first_class.head(5) 
qty = (len(first_class))
qty = 'There are {} male survivors from the first Class'.format(qty)
print(qty) 


# #28 Quantos homens da terceira classe sobreviveram?

# In[449]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3)]
first_class = ffc_sur[first_class.columns[1:4]]
first_class.head(5) 
qty = (len(first_class))
qty = 'There are {} male survivors from the first Class'.format(qty)
print(qty) 


# In[ ]:





# #29 Quantas garotas da primeira classe sobreviveram?

# In[453]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18)]
qty = (len(ffc_sur))
qty = 'There are {} female survivors under the age of 18 from the First Class'.format(qty)
print(qty)     
ffc_sur
ffc_sur = (len(first_class))


# #30 Quantas garotas da segunda classe sobreviveram?

# In[454]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18)]
qty = (len(ffc_sur))
qty = 'There are {} female survivors under the age of 18 from the Second Class'.format(qty)
print(qty)     
ffc_sur
ffc_sur = (len(first_class))


# #31 Quantas garotas da terceira classe sobreviveram?

# In[455]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18)]
third_class = ffc_sur
third_class.head(5) 
qty = (len(second_class))
qty = 'There are {} female  survivors from the Third Class'.format(qty)
print(qty)     


# #32 Quantas garotos da primeira classe sobreviveram?

# In[456]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18)]
first_class = ffc_sur
qty = (len(first_class))
qty = 'There are {} male survivors under the age of 18  from the First Class'.format(qty)
print(qty)  


# #33 Quantas garotos da segunda classe sobreviveram?

# In[457]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18)]
second_class = ffc_sur
qty = (len(second_class))
qty = 'There are {} male survivors under the age of 18  from the Second Class'.format(qty)
print(qty) 


# In[458]:


#34 Quantas garotos da terceira classe sobreviveram?


# In[459]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18)]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} male survivors under the age of 18  from the Third Class'.format(qty)
print(qty)


# #35 Vamos visualizar os sobreviventes por idade

# In[460]:


facet = sns.FacetGrid(df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df['Age'].max()))
facet.add_legend()


# #36 Visulizar ticket médio por Classe

# In[529]:


fig, axis1 = plt.subplots(1,1,figsize=(18,12)) 
average_age = df[["Fare","Pclass"]].groupby(['Pclass'],as_index=False).mean()
sns.barplot(x='Fare', y='Pclass', data=average_age) 


# #37 Visualizar ticket médio por genero

# In[530]:


fig, axis1 = plt.subplots(1,1,figsize=(14,8)) 
average_age = df[["Fare","Sex"]].groupby(['Sex'],as_index=False).mean()
sns.barplot(x='Fare', y='Sex', data=average_age)


# #38 Find Adults Females that survived from the 1st Class that board in Queenstown 

# In[463]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)


# #39 Find Adults Females that survived from the 1st Class that board in Cherborough 

# In[464]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 1st Class that board in Cherborough '.format(qty)
print(qty)


# #40 Find Adults Females that survived from the 1st Class that board in Southhampton 

# In[465]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)


# In[466]:


#41 How many Adults Females survived from the 2nd Class that board in Queenstown?


# In[467]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)


# #42 How many Adults Females that survived from the 2nd Class board in Cherborough?

# In[468]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)


# #43 How many Adults Females that survived from the 1st Class board in Queenstown?

# In[469]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)


# #44 How many Adults Females that survived from the 3rd Class  board in Queenstown?

# In[470]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)


# #45 How many Adults Females that survived from the 3rd Class board in Cherborough?

# In[471]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 3rd Class that board in Cherborough '.format(qty)
print(qty)


# #46 How many Adults Females that survived from the 3rd Class that board in Queenstown?

# In[472]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 3rd Class that board in Southampton '.format(qty)
print(qty)


# #47 How many Adults Males that survived from the 1st Class board in Queenstown?

# In[473]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)


# #48 How many Adults Males that survived from the 1st Class board in Cherborough?

# In[474]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 1st Class that board in Cherborough '.format(qty)
print(qty)


# #49 How many Adults Males that survived from the 1st Class board in Southhampton?

# In[475]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)


# #50 How many Adults Males that survived from the 2nd Class board in Queenstown?

# In[476]:


ffc_sur = df[(df.Sex == 'Male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)


# #51 How many Adults Males that survived from the 2nd Class board in Queenstown? 

# In[477]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)


# #52 How many Adults Males that survived from the 1st Class board in Queenstown? 

# In[478]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)


# #53 How many Adults Males that survived from the 3rd Class board in Queenstown?

# In[479]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)


# #54 How many Adults Males that survived from the 3rd Class board in Cherborough?

# In[480]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 3rd Class that board in Cherborough '.format(qty)
print(qty)


# #55 How many Adults Males that survived from the 3rd Class board in Queenstown?

# In[481]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 3rd Class that board in Southampton '.format(qty)
print(qty)


# 
# # CHILDREN

# # 1st Class
# #GIRLS 

# #56 How many girls that survived from the 1st Class board in Queenstown? 

# In[482]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)


# #57 How many girls that survived from the 2nd Class board in Cherborough?

# In[483]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)


# #58 How many girls that survived from the 1st Class that in Queenstown? 

# In[484]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)


# # 2nd
# #59 How many girls that survived from the 2nd Class board in Queenstown?

# In[485]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)


# #60 How many girls that survived from the 2nd Class board in Cherborough?

# In[486]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)


# #61 How many girls that survived from the 2nd Class board in Southhampton?

# In[487]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)


# # 3rd
# #62 How many girls that survived from the 2nd Class that board in Queenstown? 

# In[488]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)


# #63 How many girls that survived from the 2nd Class board in Cherborough?

# In[489]:


ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 3rd Class that board in Cherborough '.format(qty)
print(qty)  


# In[490]:


#64 How many girls that survived from the 2nd Class board in Queenstown?


# In[491]:



ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 3rd Class that board in Southampton '.format(qty)
print(qty)


#  # CHILDREN  - BOYS
# 
# # 1st
# 

# #65 How many boys that survived from the 1st Class board in Queenstown?

# In[492]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)


# #66 How many boys that survived from the 1st Class that in Cherborough?

# In[493]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 1st Class that board in Cherborough '.format(qty)
print(qty)


# #67 How many boys that survived from the 1st Class board in Queenstown?

# In[494]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)


# # 2nd
# #68 How many boys that survived from the 2nd Class board in Queenstown? 

# In[495]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)


# #69 How many boys that survived from the 2nd Class that in Cherborough?

# In[496]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)


# #70 How many boys that survived from the 2nd Class board in Southhampton?

# In[497]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)


# # 3rd
# #71 How many boys that survived from the 2nd Class board in Queenstown? 

# In[498]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)


# In[499]:


#72 How many boys that survived from the 2nd Class that in Cherborough?


# ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'C')]
# third_class = ffc_sur
# qty = (len(third_class))
# qty = 'There are {} boys that survived from the 3rd Class that board in Cherborough '.format(qty)
# print(qty)

# #73 How many boys that survived from the 2nd Class board in Queenstown?

# In[500]:


ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 3rd Class that board in Southampton '.format(qty)
print(qty)


# #74 Qual a idade média dos sobreviventes do naufrágio?
# #75 Qual a idade média das vítimas do naufrágio?

# In[517]:


df.groupby("Survived").agg({"Age" : np.mean})


# In[518]:


#76 Qual a idade do sobrevivente mais novo?
#77 Qual a idade da vítima mais nova?


# In[519]:


df.groupby('Survived').agg({'Age' : np.min})


# #77 Qual a idade do sobrevivente mais idoso?
# #78 Qual a idade da vítima mais idosa?

# In[521]:


df.groupby("Survived").agg({"Age" : np.max})


# In[515]:


df.groupby("Survived").agg({"Age" : np.max})


# In[ ]:


# Quais fatores contribuíram para a sobrevivência dos passageiros a bordo?


# In[ ]:





# In[ ]:




