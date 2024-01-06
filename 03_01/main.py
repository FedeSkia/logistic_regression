import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

loan = pd.read_csv('data/loan.csv')

'''
We want to predict if a borrower will default or not default 
a new loan based on their income and amount of money
default is the response variable while income and loan amount are the predictors
or independent variables or features
'''

#loan.info()
#print(loan.describe())

#ax = sns.boxplot(data=loan, x='Default', y='Income')
#plt.show()

#ax = sns.boxplot(data=loan, x='Default', y='Loan Amount')
#plt.show()

''' Lets create a scatterplot that describes the relationship between the annual income
 and loan outcomes '''
default_as_number = np.where(loan['Default']=='No', 0, 1) #if no == 0 otherwise its 1
#ax = sns.scatterplot(x=loan['Income'], y=default_as_number, s=150)
#plt.show()

''' Lets create a scatterplot that describes the relationship between the amount borrowed
 and loan outcomes '''
#ax = sns.scatterplot(x=loan['Loan Amount'], y=default_as_number, s=150)
#plt.show()

'''
Lets create the test and train datasets
'''
y = loan['Default'] #dependant variable
X = loan[['Income', 'Loan Amount']] #independent variable

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=123)

''' Lets train and evaluate the model '''
classifier = LogisticRegression()
model = classifier.fit(X_train, y_train)

print(model.predict(X_test))

''' To evaluate how accurate our model is we pass the test data (X_test and y_test) to the score method '''
print("Model score", model.score(X_test, y_test))
'''The output of the score (0.89) tells that it is able to predict 89% of the label in the test set
The accuracy of a model only gives us a one dimensional perspective of performance. 
To get a broader perspective we need to generate a confusion matrix of model's perfomance 
'''

print("Confusion matrix", confusion_matrix(y_test, model.predict(X_test)))
'''
[[3 1]    [[TP FP]
 [0 5]]    [FN TN]]
'''

'''
Lets interpret the model.
We want to understand the model coefficients
To get the intercepton (B_0) we do model.intercept_
to get the coefficients (B_1 and B_2) we do model.coef_ 
'''
print("Model intercept", model.intercept_) #15.4670632
print("Model coef", model.coef_) # [-1.0178107   0.14656096] [Income, LoanAmount]

''' 
Since income coef is negative we can say infer that the chances to default == Y == 1 decreases
with the increase of the income.
Since LoanAmount is positive we can infer that the chances to default == Y == 1 increase the probability
of a default 
Lets calculate the odds by exponentiation the coefficients
'''

log_odds = np.round( model.coef_[0], 2) #array -> [-1.02, 0.15]
pd.DataFrame( {'log_odds': log_odds}, index = X.columns ) #Create a data frame where the index are the colunns of the X Dataframe (check above)
#           Log Odds
#Income     -1.02
#LoanAmount 0.15

odds = np.round( np.exp(log_odds), 2) #lets round to 2nd decimal and exponentiate the dataframe
print("Dataframe with Odds", pd.DataFrame({'odds': odds}, index=X.columns))

'''
The dataframe with odds looks like 
            odds
Income       0.36   Means that ( 1 - 0.36 ) = 64% means that for every 1$ of income the odds that they will default on their loans reduces by 64% 
Loan Amount  1.16   Means that ( 1.16 - 1 ) = 16% means that for every 1$ of the amount borrowed the odds that will default on their loans increases by 16%
'''

