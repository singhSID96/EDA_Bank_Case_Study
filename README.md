# EDA_Bank_Case_Study
About In this case study ,the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.
In this case study ,the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.

Business Understanding : The loan providing companies find it hard to give loans to the people due to their insufficient or non-existent credit history. Because of that, some consumers use it as their advantage by becoming a defaulter. Suppose you work for a consumer finance company which specialises in lending various types of loans to urban customers. You have to use EDA to analyse the patterns present in the data. This will ensure that the applicants capable of repaying the loan are not rejected. When the company receives a loan application, the company has to decide for loan approval based on the applicant’s profile.

Business Objectives : The company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default. The company can utilise this knowledge for its portfolio and risk assessment.

Download the dataset from the link given as below. 
https://drive.google.com/open?id=16RQztUqCfJOlbooHqYlJrp6Q7iL65uZB

Here is some sample code in Python that can be used to apply Association Rule Mining in order to identify relationships between independent continuous variables and dependent continuous variables:

#import the necessary packages
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori

#load the data
data = pd.read_csv('data.csv')

#convert the data into a form suitable for the apriori algorithm
transaction_data = []
for i in range(0, data.shape[0]):
    temp = []
    for j in range(0, data.shape[1]):
        temp.append(str(data.values[i,j]))
    transaction_data.append(temp)

#apply the apriori algorithm to the data
rules = apriori(transaction_data, min_support = 0.3, min_confidence = 0.8, min_lift = 3, min_length = 2)

#output the rules
print(rules)
