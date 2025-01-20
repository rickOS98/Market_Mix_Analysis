#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Import the Excel Package

# In[2]:


pip install xlrd


# ### Load The Data Set

# In[3]:


data = pd.read_excel("C:\\Users\\captr\\OneDrive\\Desktop\\marketing data.xls")
data.head()


# ### Data Inspection

# In[4]:


data.Dt_Customer.head()


# In[5]:


print(data.columns)


# In[6]:


data[' Income '].head()


# In[7]:


data.isnull().sum()


# ### As we can see that the income column has 24 null values so we will fix it

# In[8]:


data.columns = data.columns.str.strip()
data['Income'] = data['Income'].replace('[\$,]', '', regex=True).astype(float)
income_means = data.groupby(['Education', 'Marital_Status'])['Income'].transform('mean')
data['Income'] = data['Income'].fillna(income_means)


# In[9]:


data.head(30)


# In[10]:


data.isnull().sum()


# ### Woah !! Problem solved as you can see from above

# In[11]:


type(data.Kidhome)


# ### Creating  variables to represent the total number of children

# In[12]:


data["total_Children"] = data.Kidhome+data.Teenhome
data.head()


# In[13]:


from datetime import datetime



# ### Creating variables to represent the age of the customer

# In[14]:


current_age = datetime.now().year
data['Age'] = current_age - data['Year_Birth']
data.head()


# ### Creating variables to represent the total spending of the customer

# In[15]:


spending_cols=["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]


# In[16]:


data["Total_Spending"]=data[spending_cols].sum(axis=1)
data.head(5)


# In[17]:


data["Total_Spending"]=data.MntWines + data.MntFruits + data.MntMeatProducts + data.MntFishProducts + data.MntSweetProducts + data.MntGoldProds


# In[18]:


data.head()


# ### Creating variables to represent the total Purchasing of the customer

# In[19]:


data["Total_Purchases"]=data.NumDealsPurchases + data.NumWebPurchases + data.NumCatalogPurchases + data.NumStorePurchases
data.head()


# ### Generated box plots and histograms to gain insights into the distributions and identify outliers.

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


import matplotlib.pyplot as plt

# Selecting only the numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns


for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    plt.hist(data[col], bins=20, alpha=0.5, color='blue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[22]:


numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
plt.hist(numeric_columns,bins=20,alpha=0.5,density=True,histtype="stepfilled",color="black")


# In[23]:


numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    data.boxplot(column=col)
    plt.title(f'Box Plot of {col}')


# ### Applied ordinal and one-hot encoding based on the various types of categorical variables

# In[24]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# In[25]:


ordinal_columns = ['Education'] 
nominal_columns = ['Marital_Status','Country'] 


# In[26]:


ordinal_encoder = OrdinalEncoder()
data[ordinal_columns] = ordinal_encoder.fit_transform(data[ordinal_columns])


# In[27]:


onehot_encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity
nominal_encoded = onehot_encoder.fit_transform(data[nominal_columns])


# In[28]:


nominal_encoded_df = pd.DataFrame(nominal_encoded, columns=onehot_encoder.get_feature_names_out(nominal_columns))


# In[29]:


data = pd.concat([data, nominal_encoded_df], axis=1)
data.drop(columns=nominal_columns, inplace=True)


# In[30]:


data.head()


# ### Generated a heatmap to illustrate the correlation between different pairs of variables

# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns for the correlation matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# correlation matrix
correlation_matrix = numeric_data.corr()

# heatmap
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()



# # Hypothesis Testing 

# ### i) Older individuals may not possess the same level of technological proficiency and may, therefore, lean toward traditional in-store shopping preferences

# In[33]:


import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


age_instore_corr, age_instore_pval = stats.spearmanr(data['Age'], data['NumStorePurchases'])
age_online_corr, age_online_pval = stats.spearmanr(data['Age'], data['NumWebPurchases'])

print(f"Spearman correlation between Age and In-store Purchases: {age_instore_corr:.2f}, p-value: {age_instore_pval:.4f}")
print(f"Spearman correlation between Age and Online Purchases: {age_online_corr:.2f}, p-value: {age_online_pval:.4f}")

# Regression analysis for age and shopping preferences
instore_model = smf.ols('NumStorePurchases ~ Age', data=data).fit()
online_model = smf.ols('NumWebPurchases ~ Age', data=data).fit()

print("\nIn-store Purchases Regression Summary:\n", instore_model.summary())
print("\nOnline Purchases Regression Summary:\n", online_model.summary())


# ### ii) Customers with children likely experience time constraints, making online shopping a more convenient option.

# In[35]:


import scipy.stats as stats
import statsmodels.formula.api as smf


# Step 1: Perform Spearman's correlation test
children_online_corr, children_online_pval = stats.spearmanr(data["total_Children"], data['NumWebPurchases'])

print(f"Spearman correlation between Total Children and Online Purchases: {children_online_corr:.2f}, p-value: {children_online_pval:.4f}")

# Step 2: Regression analysis for Total Children and Online Purchases
online_model = smf.ols('NumWebPurchases ~ total_Children', data=data).fit()

print("\nOnline Purchases Regression Summary:\n", online_model.summary())


# ### iii) Sales at physical stores may face the risk of cannibalization by alternative distribution channels.

# In[36]:


import scipy.stats as stats
import statsmodels.formula.api as smf

# Step 1: Perform Spearman's correlation between store and alternative channels
store_online_corr, store_online_pval = stats.spearmanr(data['NumStorePurchases'], data['NumWebPurchases'])
store_catalog_corr, store_catalog_pval = stats.spearmanr(data['NumStorePurchases'], data['NumCatalogPurchases'])

print(f"Spearman correlation between Store and Online Purchases: {store_online_corr:.2f}, p-value: {store_online_pval:.4f}")
print(f"Spearman correlation between Store and Catalog Purchases: {store_catalog_corr:.2f}, p-value: {store_catalog_pval:.4f}")

# Step 2: Multiple regression analysis to evaluate impact on store sales
cannibalization_model = smf.ols('NumStorePurchases ~ NumWebPurchases + NumCatalogPurchases', data=data).fit()

print("\nMultiple Regression Summary for Store Purchases:\n", cannibalization_model.summary())


# # Data Visualisation

# ### Identifying the top-performing products and those with the lowest revenue.

# In[45]:


import matplotlib.pyplot as plt

# Calculate total revenue for each product
product_revenue = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()

# Sort revenues in descending order for clear visualization
product_revenue = product_revenue.sort_values(ascending=False)

# Plot the revenue for each product
plt.figure(figsize=(10, 6))
product_revenue.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Total Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()


# ### Examining if there is a correlation between customers' age and the acceptance rate of the last campaign

# In[47]:


import matplotlib.pyplot as plt
import scipy.stats as stats


from datetime import datetime
data['Age'] = datetime.now().year - data['Year_Birth']

# Step 1: Calculated acceptance rate by age group
age_response = data.groupby('Age')['Response'].mean()

# Step 2: Scatter plotted for visual analysis
plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['Response'], alpha=0.5, color='blue')
plt.title("Correlation between Age and Campaign Acceptance")
plt.xlabel("Age")
plt.ylabel("Campaign Acceptance (1 = Accepted, 0 = Not Accepted)")
plt.show()

# Step 4: Calculate Spearman correlation to measure the relationship
correlation, p_value = stats.spearmanr(data['Age'], data['Response'])
print(f"Spearman Correlation between Age and Campaign Acceptance: {correlation:.2f}, p-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("There is a statistically significant correlation between age and campaign acceptance.")
else:
    print("There is no statistically significant correlation between age and campaign acceptance.")


# ### Determining the country with the highest number of customers who accepted the last campaign

# In[52]:


plt.figure(figsize=(10, 6))
plt.scatter(data['total_Children'], data['Total_Spending'], alpha=0.5, color='green')
plt.title("Relationship between Number of Children and Total Expenditure")
plt.xlabel("Total Number of Children at Home")
plt.ylabel("Total Expenditure")
plt.show()


# ### Investigating if there is a discernible pattern in the number of children at home and the total expenditure

# In[55]:


correlation, p_value = stats.spearmanr(data['total_Children'], data['Total_Spending'])
print(f"Spearman Correlation between Total Children and Total Expenditure: {correlation:.2f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a statistically significant relationship between the number of children at home and total expenditure.")
else:
    print("There is no statistically significant relationship between the number of children at home and total expenditure.")


# ### Analyzed the educational background of customers who lodged complaints in the last two years

# In[56]:


import matplotlib.pyplot as plt


complaint_data = data[data['Complain'] == 1]


education_complaints = complaint_data['Education'].value_counts()


plt.figure(figsize=(8, 5))
education_complaints.plot(kind='bar', color='coral', edgecolor='black')
plt.title("Educational Background of Customers with Complaints")
plt.xlabel("Education Level")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.show()


print("Education Background Counts for Complaints:\n", education_complaints)

