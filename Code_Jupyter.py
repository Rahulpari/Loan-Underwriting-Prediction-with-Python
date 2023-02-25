#!/usr/bin/env python
# coding: utf-8

# # Understanding the problem statement 
# ##### Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.
# 
# # Goal:
# ##### To predict if the loan application must be approved or not (Binary classification)
# 
# # Hypothesis
# 
# ##### Salary: Applicants with high income should have more chances of loan approval
# 
# ##### Loan amount: Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high
# 
# ##### Previous history: Applicants who have repayed their previous debts should have higher chances of loan approval
# 
# ##### Loan term: Loan for less time period and less amount should have higher chances of approval
# 
# ##### EMI: Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval
# 

# #### Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np #For Mathematic calulcation
import seaborn as sns #For data visualization
import matplotlib.pyplot as plt #For plotting graph
import os #For changing directory
import sklearn #Scikit-Learn for Model Building


# #### Set up directory

# In[2]:


path1="C:/Users/Lenovo/Dropbox/0_PLACEMENT/DataScience_Portfolio/Python_Loan_Prediction"
os.chdir(path1) #Directory path1 is set


# #### Importing dataset 

# In[3]:


train = pd.read_csv("train.csv")


# In[4]:


test = pd.read_csv("test.csv")


# In[5]:


train_org = train.copy()


# In[6]:


test_org = test.copy()


# # Understanding data

# ## Exploring dataset

# In[7]:


train.columns


# In[8]:


train.shape


# In[9]:


train.dtypes


# In[10]:


test.columns


# In[11]:


test.shape


# In[12]:


test.dtypes


# ##### Interpretation: Loan_Status is the Target variable

# # Frequency

# ## Target variable

# In[13]:


train["Loan_Status"].value_counts()


# In[14]:


train["Loan_Status"].value_counts(normalize=True)


# ## Plotting Target Variable

# In[15]:


train["Loan_Status"].value_counts().plot.bar(title = "Loan_Status")
# Interpretation: Majority of the loan application is approved


# In[16]:


train["Loan_Status"].value_counts(normalize=True).plot.bar(title = "Loan_Status")
# Interpretation: 69% of the loan application is approved


# ## Plotting independent Categorical variable

# In[17]:


train["Gender"].value_counts(normalize=True).plot.bar(title = "Gender")
# Interpretation: 80% of the loan is applied by Males


# In[18]:


train["Self_Employed"].value_counts(normalize=True).plot.bar(title = "Self_Employed")
# Interpretation: 15% of the applicants are self-employed


# In[19]:


train["Married"].value_counts(normalize=True).plot.bar(title = "Married")
# Interpretation: Majority of the loan is applied by Married person


# In[20]:


train["Credit_History"].value_counts(normalize=True).plot.bar(title = "Credit_History")
# Interpretation: >80% of applicants has repaid debt


# ## Plotting independent Ordinal variable

# In[21]:


train["Dependents"].value_counts(normalize=True).plot.bar(title = "Dependents")
# Interpretation: More than half of the application has no dependents


# In[22]:


train["Education"].value_counts(normalize=True).plot.bar(title = "Education")
# Interpretation: Majority of the applicants are graduates


# In[23]:


train["Property_Area"].value_counts(normalize=True).plot.bar(title = "Property_Area")
# Interpretation: More than half of the application has no dependents


# In[24]:


train["Loan_Amount_Term"].value_counts(normalize=True).plot.bar(title = "Loan_Amount_Term")
# Interpretation: >80% of the loans has term of 360


# ## Plotting independent Numerical variable

# In[25]:


sns.displot(train["ApplicantIncome"])
# Interpretation: Majority of the applicants have income of 5000 
# The data 'Applicant Income' is not normally distributed. It is positively skewed
# The highest value is more then 80K


# In[26]:


train["ApplicantIncome"].plot.box()
# Interpretation: 'Applicant Income' shows the presence of outlier.
# This must be because of the income gap because of education. Graduate will have more outliers as compared to non-graduates


# In[27]:


# For more clarification, segregating the Income by Education
train.boxplot(column = "ApplicantIncome", by = "Education")


# In[28]:


sns.displot(train["CoapplicantIncome"])


# In[29]:


train["CoapplicantIncome"].plot.box()


# In[30]:


sns.displot(train["LoanAmount"])
#Interpretation: The loan amount is fairly normally distributed


# In[31]:


train["LoanAmount"].plot.box()
# The Loan Amount shows a lot of Outliers which needs to be treated.


# # Categorical Independent Variable vs Target Variable

# In[32]:


train.groupby("Gender")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# In[33]:


train.groupby("Married")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# In[34]:


train.groupby("Dependents")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# In[35]:


train.groupby("Education")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# In[36]:


train.groupby("Self_Employed")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# # Testing the Hypothesis
# ##### Salary: Applicants with high income should have more chances of loan approval
# 
# ##### Loan amount: Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high
# 
# ##### Previous history: Applicants who have repayed their previous debts should have higher chances of loan approval
# 
# ##### Loan term: Loan for less time period and less amount should have higher chances of approval
# 
# ##### EMI: Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval
# 

# # Numerical Independent Variable vs Target Variable

# ### Hypothesis 01: Applicants with high income should have more chances of loan approval

# In[37]:


train.groupby("Loan_Status")["ApplicantIncome"].mean().plot.bar()


# ##### Interpretation: The mean income is same for both approved and unapproved loan application
# ##### Thus, we will be dividing applicant income into buckets/bins comparing it with the approval rate

# In[38]:


# Creating income buckets/bins for applicants
bins = [0, 2500, 4000, 6000, 81000]
Group = ["Low","Average","High","Peak"]
train["Bin"] = pd.cut(train["ApplicantIncome"], bins, labels=Group)

train.groupby("Bin")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# #### Interpretation: Applicant Income does not affect the Loan approval outcome, Disproving the first hypothesis

# In[39]:


# Creating income buckets/bins for co-applicants
bins = [0, 1000, 3000, 42000]
Group = ["Low","Average","High"]
train["Bin"] = pd.cut(train["CoapplicantIncome"], bins, labels=Group)

train.groupby("Bin")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# ##### Interpretation: From the data, it seems that the copplicant Income affects the Loan approval outcome. But this cannot be true. It may be because majority of the coapplicants income data is missing as seen in the distribution plot. 
# ##### Thus, comparing the total income with approval status

# In[40]:


train["Total"] = train["ApplicantIncome"] + train["CoapplicantIncome"]

# Checking the variation
sns.displot(train["Total"])


# In[41]:


# Creating income buckets/bins for total
bins = [0, 2500, 4000, 6000, 81000]
Group = ["Low","Average","High","Peak"]
train["Bin"] = pd.cut(train["Total"], bins, labels=Group)

train.groupby("Bin")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# ##### Interpretation: low income group has low acceptance rate conmparatively as expected

# ### Hypothesis 02: Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high

# In[42]:


# Creating income buckets/bins for loan amount
bins = [0, 100, 200, 700]
Group = ["Low","Average","High"]
train["Bin"] = pd.cut(train["LoanAmount"], bins, labels=Group)

train.groupby("Bin")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# ##### It can be seen that Lower the loan Amount, higher is the chances of loan approval. Thus, Hypothesis 02 is true

# In[43]:


#Converting Target Vaiable - Loan Status to numeric value for comparing correltion with predictor variables in Heat Map
train["Loan_Status"].replace("Y", 1, inplace = True)
train["Loan_Status"].replace("N", 0, inplace = True)

#Also, converting 3+ values of dependent variable to 3
train["Dependents"].replace("3+", 3, inplace = True)
test["Dependents"].replace("3+", 3, inplace = True)

train.groupby("Dependents")["Loan_Status"].value_counts(normalize=True).unstack("Loan_Status").plot.bar(stacked=True)


# ## Heat Map
# ##### NOTE: Darker the colour, more is the correlation

# In[44]:


# Drop Total & Bin 
train = train.drop(["Bin", "Total"], axis=1)

matrix = train.corr()
ax = plt.subplots(figsize=(9, 6))
sns.heatmap (matrix, vmax=1, square=True, cmap="BuPu");


# #### High correlation is seen between (Loan Status x Credit History) & (Applicant Income x Loan Amount) 

# # Treating Missing Values

# In[45]:


# Finding missing values
train.isnull().sum()


# In[46]:


#Since the missing values are not much, filling categorical & ordinal with MODE
# NOTE: We use [0] with mode() because mode() returns a series of value and we take the first one with [0]
train["Gender"].fillna(train["Gender"].mode()[0], inplace = True)
train["Married"].fillna(train["Married"].mode()[0], inplace = True)
train["Dependents"].fillna(train["Dependents"].mode()[0], inplace = True)
train["Self_Employed"].fillna(train["Self_Employed"].mode()[0], inplace = True)
train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mode()[0], inplace = True)
train["Credit_History"].fillna(train["Credit_History"].mode()[0], inplace = True)

#Filling numerical with MEDIAN
train["LoanAmount"].fillna(train["LoanAmount"].median(), inplace = True)

train.isnull().sum()


# # Treating Outliers Values with LOG transformation

# ##### The loan Amount is Positive or right skewed. One way to remove the skewness is by doing the log transformation. As we take the log transformation, it does not affect the smaller values much, but reduces the larger values. So, we get a distribution similar to normal distribution.

# In[47]:


#Log transformation on Train data
train["LoanAmount_log"] = np.log(train["LoanAmount"])
sns.displot(train["LoanAmount_log"])

#Similarly on test data
test["LoanAmount_log"] = np.log(test["LoanAmount"])


# # Model building Type 01: Logistic Regression by Scikit Learn

# In[48]:


# Dropping the Loan ID varaible as it does not affect target variable
train = train.drop("Loan_ID", axis=1)
test = test.drop("Loan_ID", axis=1)

train.dtypes


# In[49]:


test.dtypes


# ##### Logistic regression is applied with Scikit Learn (sklearn) library of Python
# ##### For sklearn, we need target variable in a separate dataset

# In[50]:


x = train.drop("Loan_Status", axis = 1)
y = train.Loan_Status


# In[51]:


print(x)


# In[52]:


print(y)


# ##### Converting all categorical variables to dummy varaibles using get_dummies()

# In[53]:


x = pd.get_dummies(x)
train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[54]:


print(x)


# ## Modelling

# In[55]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold

#Split the Training set to Training & Validation set
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3)


# In[56]:


model = LogisticRegression(C=1e9, class_weight=None, dual=False,
fit_intercept=True, intercept_scaling=1, max_iter=100,
solver='liblinear', random_state=0)
model.fit(x_train, y_train)
x_val.dtypes


# ## Evaluation Metrics

# In[57]:


# Predicting validation data
pred_val = model.predict(x_val)


# In[58]:


#print the regression coefficients
print("The intercept b0 =", model.intercept_)
print("The coefficient b1 = \n", model.coef_)


# In[59]:


ConfusionMatrix = confusion_matrix(y_val, pred_val)
print(ConfusionMatrix)
a = sns.heatmap(ConfusionMatrix, annot=True, cmap="YlGnBu")
a.set_xlabel("PREDICTED")
a.set_ylabel("ACTUAL");
a.xaxis.set_ticklabels(["N","Y"])
a.yaxis.set_ticklabels(["N","Y"])


# In[60]:


TP = ConfusionMatrix[1,1] #True positive
TN = ConfusionMatrix[0,0] #True negative
FN = ConfusionMatrix[1,0] #True positive
FP = ConfusionMatrix[0,1] #True negative
Total=len(y_val)
print("TP =", TP)
print("TN =", TN)
print("FN =", FN)
print("FP =", FP)
print("Total =", Total)


# #### Accuracy: how many observations, both positive and negative, were correctly classified
# ##### Accuracy = (TP + TN)/Total
#     
# #### Precision: How much were correctly classified as positive out of all positives
# ##### Precision = TP/TP+FP (Positives)
# 
# #### Recall/Sensitivity: Ratio between How much were correctly identified as positive to actual total positive
# ##### Recall = TP/FN+TP (Actual Positives)
# 
# #### Specificity: Ratio between how much were correctly classified as negative to actual total negative
# ##### Specificity = TN/FP+TN (Actual Negatives)
# 
# #### F1 score: Harmonic mean of precision and recall
# ##### F1 Score = 2 * (precision * recall)/ (precision + recall)
# ##### F1 score is considered a better indicator of the classifier’s performance than the regular accuracy measure
# 
# #### ROC curves is a graphical way to show the connection/trade-off between TP (sensitivity) and FP (specificity)
# #### AUC represents the probability of model to classify positive and negative class correctly
# ##### Eg: AUC = 0.7; The model has 70% chance of correctly classifying the target variable
# ##### Eg: AUC = 0.5; worst; The model has no discrimination capacity to distinguish between class
# ##### Eg: AUC = 0; The model is reciprocating the classes - Predicting a negative class as a positive class & vice versa

# In[61]:


# Evaluation metric - Accuracy
print("Accuracy from confusion matrix is ", (TN+TP)/Total*100)
print("Accuracy from scikit learn model=",accuracy_score(y_val, pred_val)*100)


# In[62]:


# Other evaluation metrics - Precision, Recall, F1 Score
print(classification_report(y_val, pred_val, target_names=['No', 'Yes']))


# In[63]:


# ROC (AUC)
# Receiver Operating Characteristic curve - Area under Curve
y_pred_proba = model.predict_proba(x_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  y_pred_proba)

auc = roc_auc_score(y_val, y_pred_proba)
a = plt.plot(fpr,tpr,label="ROC for Validation Data, AUC="+str(auc))
plt.xlabel("FP (1 - specificity)")
plt.ylabel("TP (sensitivity)")
plt.legend(loc=4)
plt.show()


# ## Predicting Test data

# In[64]:


test.isnull().sum()


# In[65]:


#Since the missing values are not much, filling categorical & ordinal with MODE
test["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mode()[0], inplace = True)
test["Credit_History"].fillna(train["Credit_History"].mode()[0], inplace = True)

#Filling numerical with MEDIAN
test["LoanAmount"].fillna(train["LoanAmount"].median(), inplace = True)
test["LoanAmount_log"].fillna(train["LoanAmount_log"].median(), inplace = True)

test.isnull().sum()


# In[66]:


# Predicting test data
pred_test = model.predict(test)


# In[67]:


# Checking the format of submission table
submission = pd.read_csv("sample_submission.csv")
print(submission)


# In[68]:


# Saving Loan Status for test data in submission table
submission["Loan_Status"] = pred_test
submission["Loan_ID"] = test_org["Loan_ID"]
print(submission)


# In[69]:


#Converting Target Vaiable - Loan Status back to Y, N
submission["Loan_Status"].replace(1, "Y", inplace = True)
submission["Loan_Status"].replace(0, "N", inplace = True)
print(submission)

