# Project 4 - UDACITY_NANODEGREE_FUNDAMENTALS_OF DATA_SCIENCE
# Quais foram os fatores que fizeram com que algumas pessoas fossem mais propensas a sobreviver?

# Load libraries
 
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
 

import plotly.offline as py

import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Importing Dataset
df = pd.read_csv('train.csv')

# Checking rows and columns
df.shape

# Lets visualize  the first 5 entries of data
df.head()

# Let's clean the data since we have lots of missing information
df.isnull().tail()
df.isnull().sum()

# Dropping rows where Age column is missing
df.dropna(subset=['Age'], inplace=True) 
df.shape

# Lets discover some information about the dataset
# Passengers who embarked at Southhampton, Queenstown and Cherbourg
df['Embarked'].value_counts()
df['Embarked'].value_counts().sum()

# Dropping those who didn't embarked
df.dropna(subset=['Embarked'], inplace=True) 
df.shape

# Convert Column "Age" into integer
df['Age'] = df['Age'].astype(int)
df.head()

# Convert Column "Age" into integer
df['Age'] = df['Age'].astype(int)
df.head()

# Finding numbers of adults
adults  = df[(df.Age > 18)].count().astype(int)
adults = adults['Age']
adults = 'There were {}  adults'.format(adults)
print(adults)

# Finding numbers of children
adults  = df[(df.Age <= 18)].count().astype(int)
adults = adults['Age']
adults = 'There were {}  children'.format(adults)
print(adults)

survivors = df[(df.Survived == 1)].count()
survivors = survivors['Survived']
survivors = 'There were {} survivors'.format(survivors)
print(survivors)

# Finding numbers of men and women
counts = df['Sex'].value_counts()
print(counts)
print('n\There were 453 Men and 259 women aboard, including children')


# How many lives were saved?
survivors = df[(df.Survived == 1)].count()
survivors = survivors['Survived']
survivors = 'There were {} survivors'.format(survivors)
print(survivors)

# How many lives were lost?
df_lost = df.loc[df['Survived'] == 0].count().astype(int)
df_lost = df_lost['Survived']
df_lost = 'There were {} lives lost'.format(df_lost) 
print (df_lost)


# How many male adults survived
df_saved  = df[(df.Sex == 'male') & (df.Age > 18) &(df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} male adults survived'.format(df_saved)
print(df_saved)

# How many female adults survived
df_saved  = df[(df.Sex == 'female') & (df.Age > 18) &(df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} female adults survived'.format(df_saved)
print(df_saved)

# How many were boys  
boys = df[(df.Sex == 'male') & (df.Age <= 18)].count().astype(int)
boys = boys['Sex']
boys = 'There were {} boys'.format(boys)
print(boys)

# How many were girls  
girls = df[(df.Sex == 'female') & (df.Age <= 18)].count().astype(int)
girls = girls['Sex']
girls = 'There were {} girls'.format(girls)
print(girls)

# How many boys were saved
df_saved = df[(df.Sex == 'male') & (df.Age <= 18) & (df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} boys saved'.format(df_saved)
print(df_saved)

# How many girls were saved
df_saved = female_count = df[(df.Sex == 'female') & (df.Age <= 18) & (df.Survived == 1)].count().astype(int)
df_saved = df_saved['Survived']
df_saved = 'There were {} girls saved'.format(df_saved)
print(df_saved)

# List females survivors from First class
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1)]
first_class = ffc_sur
first_class = ffc_sur[first_class.columns[1:5]]
first_class.head(5)
qty = (len(first_class))
qty = 'There are {} female survivors from the First Class'.format(qty)
print(qty)     
ffc_sur.head()

# List females survivors from Second class
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2)]
second_class = ffc_sur
second_class = ffc_sur[second_class.columns[1:5]]
second_class.head(5)
qty = (len(second_class))
qty = 'There are {} female survivors from the Second Class'.format(qty)
print(qty)     
ffc_sur.head()

# Printing  females survivors from Third class
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3)]
third_class = ffc_sur[first_class.columns[1:4]]
third_class.head(5) 
qty = (len(third_class))
qty = 'There are {} female survivors from the Third Class'.format(qty)
print(qty)     
ffc_sur.head()   
  






# Printing  males survivors from First class
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1)]
first_class = ffc_sur[first_class.columns[1:4]]
first_class.head(5) 
qty = (len(first_class))
qty = 'There are {} male survivors from the first Class'.format(qty)
print(qty)     
ffc_sur.head() 

# Printing  males survivors from Second class
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2)]
second_class = ffc_sur[second_class.columns[1:4]]
first_class.head(5) 
qty = (len(second_class))
qty = 'There are {} male survivors from the Second Class'.format(qty)
print(qty)     
ffc_sur.head()
 
# Printing  males survivors from Third class
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3)]
third_class = ffc_sur[third_class.columns[1:4]]
third_class.head(5) 
qty = (len(third_class))
qty = 'There are {} male survivors from the Third Class'.format(qty)
print(qty)     
ffc_sur.head() 

# KIDS

# Printing  female survivors under the age of 18 from First Class 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18)]
qty = (len(ffc_sur))
qty = 'There are {} female survivors under the age of 18 from the First Class'.format(qty)
print(qty)     
print('Listing information')
ffc_sur
ffc_sur = (len(first_class))

# Printing  female survivors under the age of 18 from Second Class 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18)]
qty = (len(ffc_sur))
qty = 'There are {} female survivors under the age of 18 from the Second Class'.format(qty)
print(qty)     
print('Listing information')
ffc_sur

# Printing  female survivors under the age of 18 from Third Class 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18)]
qty = (len(ffc_sur))
qty = 'There are {} female survivors under the age of 18 from the Third Class'.format(qty)
print(qty)     
print('Listing information')
ffc_sur
 
# Printing  male survivors under the age of 18 from First Class 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18)]
first_class = ffc_sur
qty = (len(first_class))
qty = 'There are {} male survivors under the age of 18  from the First Class'.format(qty)
print(qty) 

# Printing  male survivors under the age of 18 from First Class 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18)]
second_class = ffc_sur
qty = (len(second_class))
qty = 'There are {} male survivors under the age of 18  from the Second Class'.format(qty)
print(qty) 

# Printing  male survivors under the age of 18 from First Class 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18)]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} male survivors under the age of 18  from the Third Class'.format(qty)
print(qty)

# Lets visualize the price Fares
df['Fare'].hist(color='darkred',bins=40,figsize=(8,4))

# Let's visualize the age of passengers 
sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=40)

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df['Age'].max()))
facet.add_legend()
# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4)) 
average_age = df[["Age","Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)

# FINDING MAIN CHARACTERS

"""Finding Thomas Andrews
Born: February 7, 1873
Birthplace: Comber, County Down, Ireland
Death: April 15, 1912, Atlantic Ocean (perished in Titanic sinking)
"""
thomas= df[df['Name'].str.contains('Andrews', na = False)]
thomas = df[(df.PassengerId == 807) ]
thomas

"""John Jacob Astor
Born: July 13, 1864
irthplace: Rhinebeck, New York
Death: April 15, 1912, Atlantic Ocean (perished in Titanic disaster)
"""
john= df[df['Name'].str.contains('Astor', na = False)]
john

"""There were only one black person in the ship
The Laroche family 
Joseph Phillippe Lemercier Laroche
Url - http://www.chasingthefrog.com/reelfaces/titanic.php
"""
joseph = df[df['Name'].str.contains('Laroche', na = False)]
joseph

"""Margaret "Molly" Brown
Born: July 18, 1867
Birthplace: Hannibal, Missouri
Death: October 26, 1932, Barbizon Hotel, New York City (brain tumor) 
"""
margaret = df[df['Name'].str.contains('Tobin')]
margaret 

# BY CLASSES - FEMALES

# 1st

# Find Adults Females that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)

# Find Adults Females that survived from the 1st Class that board in Cherborough 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 1st Class that board in Cherborough '.format(qty)
print(qty)

# Find Adults Females that survived from the 1st Class that board in Southhampton 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)

# 2nd

# Find Adults Females that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)

# Find Adults Females that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)

# Find Adults Females that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)

# 3rd

# Find Adults Females that survived from the 3rd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)

# Find Adults Females that survived from the 3rd Class that board in Cherborough
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 3rd Class that board in Cherborough '.format(qty)
print(qty)

# Find Adults Females that survived from the 3rd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Females that survived from the 3rd Class that board in Southampton '.format(qty)
print(qty)

# BY CLASSES - MALES

# 1st

# Find Adults Males that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)

# Find Adults Males that survived from the 1st Class that board in Cherborough 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 1st Class that board in Cherborough '.format(qty)
print(qty)

# Find Adults Males that survived from the 1st Class that board in Southhampton 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)

# 2nd

# Find Adults Males that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'Male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)

# Find Adults Males that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)

# Find Adults Males that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)

# 3rd

# Find Adults Males that survived from the 3rd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)

# Find Adults Males that survived from the 3rd Class that board in Cherborough
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 3rd Class that board in Cherborough '.format(qty)
print(qty)

# Find Adults Males that survived from the 3rd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age > 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} Adults Males that survived from the 3rd Class that board in Southampton '.format(qty)
print(qty)

# CHILDREN  --------///-------  GIRLS

# 1st
# Find girls that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)

# Find  girls that survived from the 1st Class that board in Cherborough
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 1st Class that board in Cherborough '.format(qty)
print(qty)

# Find girls that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)

# 2nd
#Find girls that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)

# Find  girls that survived from the 2nd Class that board in Cherborough
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)

# Find girls that survived from the 2nd Class that board in Southhampton
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)

# 3rd
#Find girls that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)

# Find  girls that survived from the 2nd Class that board in Cherborough
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 3rd Class that board in Cherborough '.format(qty)
print(qty)

# Find girls that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'female') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} girls that survived from the 3rd Class that board in Southampton '.format(qty)

# CHILDREN  --------///-------  BOYS

# 1st
# Find boys that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 1st Class that board in Queenstown '.format(qty)
print(qty)

# Find  boys that survived from the 1st Class that board in Cherborough
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 1st Class that board in Cherborough '.format(qty)
print(qty)

# Find boys that survived from the 1st Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 1) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 1st Class that board in Southampton '.format(qty)
print(qty)

# 2nd
#Find boys that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 2nd Class that board in Queenstown '.format(qty)
print(qty)

# Find  boys that survived from the 2nd Class that board in Cherborough
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 2nd Class that board in Cherborough '.format(qty)
print(qty)

# Find boys that survived from the 2nd Class that board in Southhampton
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 2) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 2nd Class that board in Southampton '.format(qty)
print(qty)

# 3rd
#Find boys that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'Q')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 3rd Class that board in Queenstown '.format(qty)
print(qty)

# Find  boys that survived from the 2nd Class that board in Cherborough
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'C')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 3rd Class that board in Cherborough '.format(qty)
print(qty)

# Find boys that survived from the 2nd Class that board in Queenstown 
ffc_sur = df[(df.Sex == 'male') & (df.Survived == 1) & (df.Pclass == 3) & (df.Age <= 18) & (df.Embarked == 'S')]
third_class = ffc_sur
qty = (len(third_class))
qty = 'There are {} boys that survived from the 3rd Class that board in Southampton '.format(qty)
print(qty)


# Did the passengers who paid more for the fare had a higher chance of surviving? Or is it the contrary?
# peaks of survivors by price paid
facet = sns.FacetGrid(df, hue="Survived",aspect=5)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, df['Fare'].max()))
facet.add_legend()






