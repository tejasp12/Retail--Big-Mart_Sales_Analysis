# Big-Mart-Sales_Analysis: A Multiple Linear Regression Problem
## Domain:Retail
## How I approched the problem and built a Regression Model to predict the sales:
### Problem Statement:
  We have a data of One Thousand Five Hundred and Fifty Nine products across 10 stores of Big-Mart chain in 10 cities.The aim is to build a Predictive Model and find out the sale of each product at a perticular store.
### Value to Business:
  __"_Using this predictive model,the decision makers of the BigMart will try to understand the properties of various products and stores which play an important in optimizing their Marketing efforts and results in increased sales._"__ 

Let's first try to understand what could affect the target variable "Sales"?
1. The day of the week - Weekends are tend to more busier than weekdays
2. The time of the day - Morning or late evening
3. At the End of the Year or any special occasions
4. Store size and location
5. Items with more shelf space sell more.
---
The whole solution is divided into 8 parts namely:
1. Get the Training data.
2. Clean the dataset.
3. Explore the Training dataset and perform feature engineering.
4. Get the Testining data and do the same treatments that performed on Training dataset.
5. Build the Model.
6. Import Test Dataset.
   6.1 Perform the same treatments on Test Dataset.
7. Perform Predictions.
8. Evaluate the Model with error metrics.
9. Final words on **Improvement of Model's Predictive Power.**
---
#### 1.Get the Training data:
During importing the dataset it is very important to focus on the type of the Data File.In most cases,the data file is in the .csv format but sometimes it also in .xlx or .xlsx format also.So,it is important to keep an eye on the file extension during importing the dataset and usage of pandas function.We can import the dataset file using **Data Manipulation library named Pandas.** Pandas has several functions to read various file types including .csv,.xls or .xlsx,.json,.html,.feather,.clipboard,.pickle,.sql,.sql_query and many more.But as I said before most of the time we encounter the dataset is in either .csv format or in .xlx or .xlsx format.Following,I have written one function that identify automatically the input file extension and then apply the appropriate pandas read function.Currently,the function only identify 3 types namely .csv,.xlx and .xlsx.
```
#===========================PART--1============================================#
#        [FUNCTION PURPOSE - FINDING FORMAT OF THE INCOMMING FILE AND 
#        THEN APPLY APPROPRIATE FILE-READ FUNCTION TO READ THAT FILE]
#==============================================================================#
#A function that read the input file and return dataframe as function output::
def read_file(input_file_path):

  #FOR EXCEL FILE FORMAT-->
  #Select the file reading function based on the input file-->
  if input_file_path[-5:] == '.xlsx' or input_file_path[-4:] == '.xls':
    print("[INFO]::DATASET IS IN EXCEL FORMAT")
    print(input_file_path)
    input_df = pd.read_excel(input_file_path)
    return input_df

  #FOR CSV FILE FORMAT-->
  if input_file_path[-4:] == '.csv':
    print("[INFO]::DATASET IS IN CSV FORMAT")
    #Read file using pandas read_csv function::
    print(input_file_path)
    input_df = pd.read_csv(input_file_path,error_bad_lines=False)
    return input_df
```
> **IMPORTANT NOTE:** All the treatments that I am showing here is related to the **Training Dataset** and the same treatments you need to perform on **Testing Dataset** if Training and Testing datasets are different.If only given the Training Dataset then perform all the treatments on that dataset and then split the dataset into **Train and Test** datasets.By doing so,you don't need to perform all operations again on Testing Dataset.

#### 2.Clean the Dataset:
In this,we are going to do a lot of subtasks and these are:
* Imputation of missing values.
* Treatment for typos.
* Scaling of the variables.
* Distribution of the univariate variables and treatment for outliers.
Let's go one by one...!
##### Imputation of missing values:
