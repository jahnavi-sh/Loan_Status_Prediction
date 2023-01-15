#Understanding the problem statement -

#Consider, there is a finance company that gives loan to people. 
#But before processing a loan this company considers and evaluates 
#several parameters of the candidates. This loan eligibility process 
#is based on customer detail provided while filling out online 
#application forms. These details are gender, marital status, 
#education, number of dependents, income, loan amount, credit history
#and others. 
#to automate this process, they have provided a dataset to identify 
#the customer segments that are eligible for loan amounts so that 
#they can specifically target these customers. 

# Here, we will build a machine learning algorithm that automates 
# this process of selecting the candidates for loan based on their 
# eligibility.  

#workflow 
#1. load data 
#2. data preprocessing 
#3. train test split 
#4. model used - support vector machine model 
#5. prediction - approved or rejected 

#load libraries 

#linear algebra - construct matrices 
import numpy as np 

#data preprocessing anf exploration 
import pandas as pd 

#data visualisation
import seaborn as sns        #to see correlation between features using heatmap

#algorithms 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection 
loan_dataset = pd.read_csv(r'loan_data.csv')

type (loan_dataset)

#view dataset 

#first five rows of dataset 
loan_dataset.head()

#dataset contains the following columns 
#1. Loan_ID - loan identification number 
#2. Gender - Male of Female
#3. Married - marital status of the applicant. Yes or No 
#4. dependencies 
#5. Education - education level. Graduate or not 
#6. Self_Employed - a person is self employed if they operate a self run business. Yes or No 
#7. ApplicantIncome 
#8. CoapplicantIncome 
#9. LoanAmount 
#10.Loan_Amount_Term 
#11.Credit_History 
#12.Property_Area
#13.Loan_Status - Y or N

#view the total number of rows and columns of the dataset 
loan_dataset.show
#data has 614 rows (614 data points) and 13 columns 

#statistical measures 
loan_dataset.describe()

#finding missing values 
loan_dataset.isnull().sum()

#there are 13 missing values in 'Gender', 3 missing points in 'Married', 15 missing points in 'Dependencies', 32 missing points in 
#'Self_Employed', 22 missing points in 'LoanAmount', 14 missing points in 'Loan_Amount_Term and 15 missing points in 'Credit_History'
#this is a comparatively small value. So for now we will drop the missing values 
#another method is to replace the missing values with the mean value. But since most of the columns in this dataset are categorical we cannot 
#employ this method of replacement.
#fixing the missing values 
loan_dataset = loan_dataset.dropna()

#as all the categorical values are binary so we can use label encoder for all such columns and the values will change into int
#label encoding 
loan_dataset.replace({'loan_status':{'N':0, 'Y':1}}, inplace=True)

#dependent column values 
loan_dataset['dependent'].value_counts()

#replace all the 3+ values to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

#data visualisation 
#education and loan status 
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
#Loan approval rate for graduated people is more 

#marital status and loan status 
sns.countplot(x='Married', y='Loan_Status', data=loan_dataset)
#Loan approval rate for Married couple is more 

#we can check plot for any features in similar manner 

#convert categorical columns to numerical values
loan_dataset.replace({"Married":{'No':0,'Yes':1}, 'Gender':{"Female":1,'Male':0},'Self_Employes':{'No':0,'Yes':1}, 'Property_Area':{'Rural':0,'Semi_Urban':1,'Urban':2}, 'Education':{'Graduate':1, 'Non_Graduate':0}}, inplace=True)

#separate data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1,random_state=2, stratify=Y)

#train model 
#support vector machine model 
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

#evaluate model 
#accuracy score 
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("accuracy score for training data", training_data_accuracy)
#accuracy score is 79%

#test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("accuracy score for test data", test_data_accuracy)
#accuracy score is 83%