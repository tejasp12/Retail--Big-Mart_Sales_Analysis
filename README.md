# Big-Mart-Sales_Analysis: A Multiple Linear Regression Problem
## Domain:Retail
## Model Stage:Prototype
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
   * Perform the same treatments on Test Dataset.
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
> It is good practice to identify and replace missing values for each column in input dataset prior to training of the model.This is called **missing data imputation**, or **imputing** in short way.

We can use `train_dataset.isnull().sum()`or `train_dataset.isna().sum()`function to check number of missing values in each of the feature columns.There are various approaches for imputation of missing values and most common and very basics are:
1.If column contains categorical values and presence of missing values : Replace missing values with **Mode** value.
2.If column contains numerical values and presence of missing values : Replace missing values with either **mean** or **median** value.As **mean** is highly affect by the presence of outliers it is best to replace with **median** value.
After executing the null value check command,we get the following as an output:
```
Item_Identifier                 0
Item_Weight                  1463
Item_Fat_Content                0
Item_Visibility                 0
Item_Type                       0
Item_MRP                        0
Outlet_Identifier               0
Outlet_Establishment_Year       0
Outlet_Size                  2410
Outlet_Location_Type            0
Outlet_Type                     0
Item_Outlet_Sales               0
dtype: int64
```
And by observing the above output we can clearly say that the columns `Item_Weight` and `Outlet_Size` have missing values.Now,as I said before the basic imputation is depend on the type of the data in that respective column and we can see the type of each column using `train_dataset.dtypes` command and we get the output:
```
Item_Identifier               object
Item_Weight                  float64
Item_Fat_Content              object
Item_Visibility              float64
Item_Type                     object
Item_MRP                     float64
Outlet_Identifier             object
Outlet_Establishment_Year      int64
Outlet_Size                   object
Outlet_Location_Type          object
Outlet_Type                   object
Item_Outlet_Sales            float64
dtype: object
```
By observing the above output we can say that `Item_Weight` is a **numerical column** and `Outlet_Size` is a **categorical column.** And as I said before we can use **mean or median** strategy to impute missing values in numerical column case and **mode** value in case categorical column case. 
Using the following code we can impute the missing values for both the columns:
* For column `Item_Weight`:
```
#Fill missing values with 'mean' value of Item weights:
train_dataset['Item_Weight'] = train_dataset['Item_Weight'].fillna(train_dataset['Item_Weight'].mean())
```
* For column `Outlet_Size`:
```
#Fill missing values with mode' value of Outlet_Size:
train_dataset['Outlet_Size'] = train_dataset['Outlet_Size'].fillna(train_dataset['Outlet_Size'].mode()[0])
```
##### Treatment for typos:
After observing unique values in some categorical variables I have found that there are presence of typos.This means that,the same category is defind using 2 or 3 different variables. Let me elaborate this using following example:
By observing the data types of the variables there are presence of 6 categorical variables and these are:`Item_Fat_Content`,`Item_Type`,`Outlet_Identifier`,`Outlet_Size`,`Outlet_Location_Type`,`Outlet_Type`.Let's see the unique values in each of these variables:
```
train_dataset['Item_Fat_Content'].unique()
>>array(['Low Fat', 'Regular', 'low fat', 'LF', 'reg'], dtype=object)

train_dataset['Item_Type'].unique()
>>array(['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
       'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
       'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
       'Breads', 'Starchy Foods', 'Others', 'Seafood'], dtype=object)

train_dataset['Outlet_Identifier'].unique()
>>array(['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045',
       'OUT017', 'OUT046', 'OUT035', 'OUT019'], dtype=object)
       
train_dataset['Outlet_Size'].unique()
>>array(['Medium', nan, 'High', 'Small'], dtype=object)

train_dataset['Outlet_Location_Type'].unique()
>>array(['Tier 1', 'Tier 3', 'Tier 2'], dtype=object)

train_dataset['Outlet_Type'].unique()
>>array(['Supermarket Type1', 'Supermarket Type2', 'Grocery Store',
       'Supermarket Type3'], dtype=object)
```
By observing the output we can clearly see that the variable `Item_Fat_Content` contains only 2 unique values namely `Low Fat and Regular` but seems some typos like `low fat,LF,reg` etc.These typos represent the same categories but written wrongly.So,we need to clear these so that we can perform the further analysis.This can be done using the following command:
```
#Replacing all representations of variable 'Low Fat':
filt_training_dataset['Item_Fat_Content'] = filt_training_dataset['Item_Fat_Content'].replace(['low fat','LF'],'Low Fat')

#Replacing all representations of 'Regular':
filt_training_dataset['Item_Fat_Content'] = filt_training_dataset['Item_Fat_Content'].replace('reg','Regular')

We can see the output after doing above:
train_dataset['Item_Fat_Content'].unique()
>>array(['Low Fat', 'Regular'], dtype=object)----Represent only 2 unique categories.
```
##### Distribution of the univariate variables and treatment for outliers:
Distribution of **Item_Visibility** variable:

![download](https://user-images.githubusercontent.com/32562117/89148456-82c51500-d577-11ea-8f67-48cb90e24a0e.png)

As we can see from the above fig.that the distribution of the variable is **a non-Gaussian distribution** and as we know a good statistic for summarizing a non-Gaussian distribution sample of data is the Interquartile Range, or IQR for short.
```
#=================Removing the Outliers==================#:
#Define first Quartile range:
Q1 = train_dataset['Item_Visibility'].quantile(0.25)
#Define third Quartile range:
Q3 = train_dataset['Item_Visibility'].quantile(0.75)
#Define IQR range:
IQR = Q3 - Q1
#Remove the outliers:
filt_training_dataset = train_dataset.query('(@Q1 - 1.5*@IQR) <= Item_Visibility <= (@Q3 + 1.5*@IQR)')
```
Check the shape of the Training and newly created filtered training dataset:
```
#Check the shape of the resulting dataset:
filt_training_dataset.shape , train_dataset.shape
>>((8379, 13), (8523, 13))
```
#### Explore the Training dataset and perform feature engineering:
By exploring the dataset, we can observe that the column `Item_Visibility`contain the numerical values that shows the percentage of the visibility of the item.These numbers seems quite difficult while interpreting with them as a visibility and we can encode them as:
* Low Visibility
* Medium Visibility
* High Visibility
>After performed this transformation,we can easily encode those 3 bins using Label Encoder.
Now,we know the Min Vsibility and Max Visibility values and using the Mean Visibility values we can define the range as follows:
```
#Check for minimum value in column 'Item_Visibility':
filt_training_dataset['Item_Visibility'].min()
>>0.0

#Check for max value in column 'Item_Visibility':
filt_training_dataset['Item_Visibility'].max()
>>0.195721125

#Check for mean value in column 'Item_Visibility':
np.mean(filt_training_dataset['Item_Visibility'])
>>0.0630612878336319
```
So, we can set the logic like follows to find the range to fall categories:Low Visibility , Medium Visibility and High Visibility
```
# Modifying the values under the column Item_Visibility into categories as low_visibility,medium_visibility and high_visibility:
#If:
#   0.0 <= Item_Visibility < 0.063:
#       return 'Low_Viz'
#If:
#   0.063 <= Item_Visibility < 0.13:
#       return 'Med_Viz'
#If:
#   0.13 <= Item_Visibility <= 0.19:
#        return 'High_VIz'
>>filt_training_dataset['Item_Visibility_bins'] = pd.cut(filt_training_dataset['Item_Visibility'],[0.000,0.063,0.13,0.19],labels=['Low_Viz','Med Viz','High_Viz'])
```
After doing this,we check the null values in column `Item_Visibility` and found out there are **526** missing values present in the column and we can replace them using the category `Low_Viz` since this is the most occouring category and since this is categorical column we use the 'mode'to fill out the missing values.
##### Perform Label Encoding::
Now, we are performing Label Encoding for some categorical variables.After viewing the data typs of variables we have 4 categorical columns to encode.
> ##### When to use Label Encoding and When to use One-Hot Encoding:
* When categories in the variable are Ordinal in Nature then in that case we use **Label Encoding**.
* When categories in the variable are Nominal in Nature then in that case we use **One-Hot Encoding**
```
#Transforming 'Item_Visibility_bins':
filt_training_dataset['Item_Visibility_bins'] = label_encoder.fit_transform(filt_training_dataset['Item_Visibility_bins'])

#Transforming 'Outlet_Size':
filt_training_dataset['Outlet_Size'] = label_encoder.fit_transform(filt_training_dataset['Outlet_Size'])

#Transforming 'Outlet_Location_Type':
filt_training_dataset['Outlet_Location_Type'] = label_encoder.fit_transform(filt_training_dataset['Outlet_Location_Type'])
```
##### Perform One-Hot Encoding:
```
# Let's create dummies for 'Outlet_Type':
dummy = pd.get_dummies(filt_training_dataset['Outlet_Type'])

# Merging both Dataframes--- filt_training_dataset + dummy
filt_training_dataset = pd.concat([filt_training_dataset,dummy],axis=1)
```
After dropping the columns those One-Hot Encoded and splitting the data into Features and Labels,the resulting dataframe is looks like following:
* Training Features:

[Training Data.xlsx](https://github.com/tejasp12/Retail--Big-Mart_Sales_Analysis/files/5020691/Training.Data.xlsx)
##### Feature Scaling:
