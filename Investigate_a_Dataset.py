#!/usr/bin/env python
# coding: utf-8

# 
# # Investigation of the Dataset: No-show Medical Appointments 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# I am goint to investigate the **No-show appointments** dataset, which collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# 
# This dataset does not have many variables, so it will be necessary to make good use of all of them in order to answer as many questions as we can. Let's take a look at out Data Dictionary:
#   
# **Independent variables:**
# ><li> PatientId: Identification of a patient
# ><li> AppointmentID: Identification of each appointment
# ><li> Gender: Male or Female . Female is the greater proportion
# ><li> ScheduledDay: The day of the actuall appointment, when they have to visit the doctor.
# ><li> AppointmentDay: The day someone called/registered the appointment.
# ><li> Age: How old is the patient.
# ><li> Neighborhood: here the appointment takes place.
# 
# >**Categorical variables** *(True or False)*:
# 
# ><li> Scholarship: indicates if the patient is enrolled in Brasilian welfare Bolsa Familia.
# ><li> Hypertension: indicates if the patient has hypertension
# ><li> Diabetes: indicates if the patient has diabetes
# ><li> Alcoholism: indicates if the patient is an alcoholic
# ><li> Handcap: indicates if the patient is handicapped
# ><li> SMS_received: 1 or more messages sent to the patient
# 
# **Dependent variable:**
# 
# ><li> No-show: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up.
#     
# 
# ### Questions:
# 
# Every hospital would love to predict whether a patient will show up or not, as this would optimize appointments with the doctor allowing more patients to be attended during the day and reducing the waiting time in the hospital.
# 
#   - **Question 1:** What is the proportion of patients who showed up and did not showed up?
#   - **Question 2:** What is the patient age distribution of shows versus no_shows?
# 
#   - **Question 3:** Is there a relationship between the gender and the Appointment status?
#   - **Question 4:** Is there a relationship between being involved in a scholarship and the status of the appointment?
#   - **Question 5:** Is there a relationship between the patient health designation and the status of the appointment?
#   - **Question 6:** Top 10 neighborhoods with the highest number of no-show ups?
#   - **Question 7:** Is there a specific day of the week in which more not show ups occur?
#   - **Question 8:** Does the waiting days period of time affect the status of the appointment?
#  
# 
#   

# <a id='wrangling'></a>
# ## Data Wrangling & Data Cleaning
# 
# **I am going to clean the data as I wrangle it**
# 
# Data loading, observations, check for cleanliness and trim / clean of the dataset. 
# 
# ### General Properties

# In[110]:


# Import statements for all of the packages that I am going to use

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')


# In[111]:


# Load data to a dataframe and visualize the first lines
df=pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# In[112]:


# Get information about the shape of the dataframe
df.shape


# - There are 110527 records and 14 columns in the dataset.

# ### Observations & Data Cleaning Process

# **Step 1:** Check the column names to see if we have to rename them:

# In[113]:


df.columns


# In[114]:


# Let's rename some columns to make it easier to read and nicer
df.rename(columns={'PatientId':'Patient_Id',
                   'AppointmentID':'Appointment_ID',
                   'ScheduledDay':'Scheduled_Day',
                   'AppointmentDay':'Appointment_Day',
                   'No-show':'No_Show'},inplace=True)


# **Step 2:** Get information about the variables in the dataframe and make changes if necessary

# In[115]:


df.info()


# As we can see in the previous query, there are **no missing values** in the dataset. 
# 
# Although, some variable type changes are necessary:
# 
# - Patient_Id to **int**
# - Scheduled_Day and Appointment_ID to **datetime**

# In[116]:


# We want to get rid of the float type and convert the Patient_ID to int64, also in
# order to operate with the '...Day' columns, we need to change its type to datetime.

df['Patient_Id']=df['Patient_Id'].astype('int64')
df['Scheduled_Day']=pd.to_datetime(df['Scheduled_Day'])
df['Appointment_Day']=pd.to_datetime(df['Appointment_Day'])

df.head(6)


# >**Create new columns** 

# - We won't be able to use the time of the appointment day, since it looks like they are marked as 00:00:00
# - It'll be helpfull for further studies if we create a new column showing how long the patient has been waiting for the medical appointment
# 

# In[117]:


# Create the new column Waiting_Days

df['Waiting_Days'] = (df.Appointment_Day-df.Scheduled_Day).dt.days
df.Waiting_Days


# In[118]:


df.head(2)


# Since the time in the Appointment_Day column doesn't show, for those appointments that took time on the same day as they weere scheduled, it shows -1 days of waiting. Therefore, we must replace the -1 with 0 days of waiting.
# 

# In[119]:


df.Waiting_Days.replace(-1,0,inplace=True)
df.head(2)


# - It will also be useful to show the weekday in which the appointment takes place, so let's create a new column for it:

# In[120]:


# Create the new column Appointment_weekday
df['Appointment_Weekday'] = (df.Appointment_Day).dt.weekday_name

#Check shows up and no shows up per weekday
df.groupby('Appointment_Weekday')['No_Show'].value_counts()


# 
# 
# **Step 3:** Check if there are any duplicated rows:

# In[121]:


sum(df.duplicated())


# - There are **no duplicated rows.**

# **Step 4:** Let's get some statistical data for each column:

# In[122]:


df.describe()


# - The average patients age is 37 years old
# - 75% of the patients received an SMS
# - The average of waiting days for the appointment can be rounded to 10 days (75% of the patients waited 14 days for the Appointment and 50% waited up to 3 days). 
# - We can already see **some errors in the data**:
# > 1. **The minimum age is -1**, therefore we will have to get rid of this row or change replace it by the mean age which we can see is 37 years. But before we do that, lets take a look at that particular patient:
# > 2. The are still **negative waiting days**, as this is impossible, we will have to fix this ploblem
# 

# **Fix the errors:**
# 
# 1. Age = -1:

# In[123]:


df.query('Age == "-1"')


# As no particular characteristic gives you any hint on how old the patient could be, **let's drop that row** and make sure there aren't any more errors on the age:

# In[124]:


# drop the row with Age = -1
df.drop(df[df['Age'] == -1].index, inplace =True)


# In[125]:


# Check if there are more Ages with error values
df.query('Age < 0')


# - There are no more rows with invalid age values.

# 2. Negative waiting days:

# In[126]:


# Check the data related to negative waiting days:

neg_waiting_days = df.query('Waiting_Days < 0')
neg_waiting_days


# In[127]:


neg_waiting_days.describe()


# All the errors (with a mean of -3 in the waiting days) happened on May 2016 and at different Neighbourhoods (except for 2 cases at the Santo Antonio Neighbourhood), 2 of the patients are handicapped and their average age is 33 years old.
# 
# - As we cannot relate the error to any of this data, we will drop these rows, since they don't represent a high percentage (0.0045%) of the total data volume.

# In[128]:


# drop the rows with negative waiting days:
df.drop(df[df['Waiting_Days'] < 0].index, inplace =True)

# Check they are dropped
df.query('Waiting_Days < 0')


# **Step 5:** Check if there are several rows for the same patient and appointment:

# In[129]:


sum(df.Patient_Id.duplicated()), sum(df.Appointment_ID.duplicated())


# - There are no dupplicated Appointment_ID values (which would be an error)
# - There are 48228 patients for whom we have data from different appointments, so let's check these out:

# In[130]:


#Create a table for all the duplicated patients
duplc_patients=df[df.Patient_Id.duplicated()==True]


# In[131]:


# select top 10 patients who scheduled appointments more than once 
dup=duplc_patients['Patient_Id'].value_counts().iloc[0:10]
print('Top 10 patients who scheduled appointments more than once:\n\n{}'.format(dup))


# In[132]:


# Get the first and last date in out dataset for which an appointment was scheduled
first_date_sch=df.sort_values(by='Scheduled_Day', ascending=True, inplace=False).Scheduled_Day.head(1)
last_date_sch=df.sort_values(by='Scheduled_Day', ascending=True, inplace=False).Scheduled_Day.tail(1)
print('First day an appointment was scheduled:{}'.format(first_date_sch))
print('Last day an appointment was scheduled:{}'.format(last_date_sch))


# In[133]:


# Get the firstand last date in out dataset for the appointments 
first_date_app=df.sort_values(by='Appointment_Day', ascending=True, inplace=False).Appointment_Day.head(1)
last_date_app=df.sort_values(by='Appointment_Day', ascending=True, inplace=False).Appointment_Day.tail(1)
print('First appointment day:{}'.format(first_date_app))
print('Last appointment day:{}'.format(last_date_app))


# Therefore, there are patients that scheduled more than 60 appointments in less than a month and a half. As there are too many of these to just take them as an error in the data, I will assume that a patient who must attend different specialists appointments, attends all of them the same day and the system creates different records for each specialist visit.

# <a id='eda'></a>
# ## Exploratory Data Analysis

# First of all, I would like to get a quick histogram view of all of the different variables and try to get some observations.

# In[134]:


df.hist(figsize=(15,15));


# **Observations:**
# 
# - The sample of young-adult patients is more representative than that of patients older than 60, reaching the peak of representation in infants.
# - A very small sample of patients is represented by alcoholic, diabetics or handicapped patients.
# - There are fewer patients who did not receive a reminder SMS for their appointment than patients who did receive one.

# Secondly, I am going to create a mask for our dependent variable so that I can see the correlations with the different independent variables.

# In[135]:


#Create a mask
shows = df['No_Show'] == 'No'
no_shows = df['No_Show'] == 'Yes'


# From now on, in order to get better insights on the data, I am going to create **contingency tables** (used in statistics to summarize the relationship between several categorical variables) using  the crosstab function to see the results in proportion.
# 
# Also, to make sure if a relationship exist or not between variables and answer the question: "is it statistically significant?" I am going to conduct a **Chi-Square Test** on each contingency tables. Therefore, if I obtain a small Chi-Square value, that means the correlation between my two variables is very little, while if I obtain a larger value, it'll mean that there is a correlation between my two variables.
# 
# >I stablish my significance level at 95%:
# 
# >- If p ≤ 0.05 --> Statistically significant - indicates strong evidence against the null hypothesis, so I reject the null hypothesis.
# >- If p > 0.05 --> Not statistically significant - indicates weak evidence against the null hypothesis, so I fail to reject the null hypothesis.

# ### Question 1: 
# >### What is the proportion of patients who showed up and did not showed up?

# In[136]:


# Count the number of patients who showed up and did not showed up 
# as well as the total number of patients
No_Show_counts=df.groupby('No_Show').count()['Patient_Id']
No_Show_total=df.count()['Patient_Id']


# In[137]:


# Get proportion of appointment shows versus no shows
showsup=No_Show_counts.iloc[0]/No_Show_total
noshowsup=No_Show_counts.iloc[1]/No_Show_total
No_Show_prop=[showsup,noshowsup]
No_Show_prop


# In[138]:


# Plot a Pie chart with Appointment status (showed up - not showed up)

labels = ['Not Showed up', 'Showed up']
sizes = [noshowsup, showsup]
colors = ['pink', 'palegreen']
explode = (0.1, 0) # only "explode" the Not Showed up slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Show status of the Appointments')
plt.axis('equal'); # Equal aspect ratio ensures that pie is drawn as a circle.


# The majority of appointment status are **show up (79.8%)**, while those marked as **not show up** represent only the **20.2%** of the total number of records.
# 

# ### Question 2: 
# >### What is the patient age distribution of shows versus no_shows?
# 

# In[139]:


#Create a contingency table: age vs. No-show to check the difference for all ages
age_vs_no_show = pd.crosstab(index = df['Age'], columns = df['No_Show'], normalize = 'index') #With normalize='index’ we normalize over each row.
age_vs_no_show.head()


# In[140]:


# Plot a histogram chart Age vs. Appointment status (showed up - not showed up)

colors = ['palegreen','pink']

age_vs_no_show.plot(kind='bar', stacked=True, color=colors, figsize=(10,5))
plt.xlabel('Age')
plt.ylabel('Show vs. No Show proportion')
plt.title('Age vs Appointment status')
L=plt.legend()
L.get_texts()[0].set_text('Showed up')
L.get_texts()[1].set_text('No Showed up')


# - We can see in the chart above that **patients whose age is above 60 take appointment more seriously** than patients under that age.
# - Patients **between 9 and 45 years old** are those that represents a **greater proportion of no showed up appointmens.** 
# 

# In[141]:


# Average ages for different show status
df.groupby('No_Show')['Age'].mean()


# The mean for patients who showed up can be rounded to 38 and for the patients that did not showed up can be rounded to 34.

# ### T-Test
# #### Age vs No shows up

# In this case, as we have calculated the means, we are goint to conduct the T-Test, which is used to compare means between two groups of a small sample size and tell you if they are different from each other and let you know if those differences could have happened by chance.

# In[142]:


# Import the package needed
import scipy


# In[143]:


# Create columns for shows and no_whows ages
shows_age = df[df.No_Show == 'No'].Age
no_shows_age = df[df.No_Show == 'Yes'].Age


# In[144]:


# Perform the T-Test
ttest_results = scipy.stats.ttest_ind(shows_age, no_shows_age)
ttest_results


# A statistical hypothesis test determines whether there is a statistically significant relationship between the data. The null hypothesis (considered the default in the test experiment) states that there is no meaningful relationship while the alternative hypothesis states the opposite.
# 
# As we can see in our output, the p-value is notoriously small. This shows evidences against the null hypothesis, therefore, I am going to reject it and assume that **there is a meaningful relationship between the age and the appointment status.**

# ### Question 3:  
# >### Is there a relationship between the gender and the Appointment status?

# In[145]:


#Create a contingency table: Gender vs. No-show
gender_vs_no_show = pd.crosstab(index = df['Gender'], columns = df['No_Show'], normalize = 'index') #With normalize='index’ we normalize over each row.
gender_vs_no_show.head()


# In[146]:


# Plot a histogram chart Gender vs. Appointment status (showed up - not showed up)

colors = ['palegreen','pink']

gender_vs_no_show.plot(kind='bar', stacked=True, color=colors, figsize=(7,5))
plt.xlabel('Gender')
plt.ylabel('Number of Appointments')
plt.title('Gender vs Appointment status')
L=plt.legend(fancybox=True, framealpha=0.5)
L.get_texts()[0].set_text('Showed up')
L.get_texts()[1].set_text('No Showed up')


# As we can see from the plot, there is not a lot of difference on the appointment status based on the patient gender.

# ### Chi-Square Test
# #### Gender vs No shows up

# In[147]:


from scipy.stats import chi2_contingency


# In[148]:


#Conduct the Chi-Square Test(chi2), showing the p (p-value), dof(degrees of freedom) and ex(expected frequencies)
chi2, p, dof,ex = chi2_contingency(gender_vs_no_show, correction=False)
chi2, p


# The p-value is greater than 0.05% (0.99%), therefore, I am going not going to assume that **there is NOT a meaningful relationship between the gender and the appointment status.**

# ### Questions 4 & 5:  
# >### Is there a relationship between being involved in a scholarship and the status of the appointment?
# >### Is there a relationship between the patient health designation and the status of the appointment?

# Our **categorical variables** are: scholarship, Hypertension, Diabetes,  Alcoholism, Handcap, SMS_received and No_Show.

# In[149]:


#Create a list with all the categorical variables
cat_var = ['Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received']

# multiple line plot
fig=plt.figure(figsize=(14,15))
for i, var in enumerate(cat_var):
    ax = fig.add_subplot(3, 3, i+1)
    # Add title and labels
    ax.set_title(var + ' vs No_Show')
    ax.set_xlabel(var)
    ax.set_ylabel('Number of Appointments')
    # Plot the proportions
    pd.crosstab(index = df[var], columns = df['No_Show'], normalize = 'index').plot(ax=ax, kind='bar', stacked=True, color=['palegreen','pink'])
    # Modify legends
    L=plt.legend(fancybox=True, framealpha=0.7, loc=3)
    L.get_texts()[0].set_text('Showed up')
    L.get_texts()[1].set_text('No Showed up');


# In general, there is not a big difference between both appointment status (show / no-show) due to the health designation. What is very remarkable, in my opinion, is that in proportion, there are **more patients who did not attend the appointment and got a reminder SMS than those who did not attend it without having received it.**

# ### Chi-Square Test
# #### Categorical variables vs No shows up

# In[150]:


#Create a list with all the categorical variables
cat_var_chi2_test = ['Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received']


#Conduct the Chi-Square Test(chi2), showing the p (p-value), dof(degrees of freedom) and ex(expected frequencies)
# multiple chi-square test
for i, result in enumerate(cat_var_chi2_test):
    chi2, p, dof,ex = chi2_contingency(pd.crosstab(index = df[result], columns = df['No_Show']), correction=False)

    print(p)


# The Chi-Square Test p values are:
# 
#     - p value for Scholarship: 3.13149499541e-22 
#     - p value for Hipertension: 2.00724179106e-32
#     - p value for Diabetes: 4.6773921245e-07
#     - p value for Alcoholism: 0.952033520537
#     - p value for Handcap: 0.112251011129
#     - p value for SMS_received: 0.0
#     
# Therefore: 
# 
# - There is a **meaningful relationship** between the independent variables **Scholarship**, **Hipertension**, **Diabetes** and **SMS_received** with our dependent variable Appointment status (No_Show)
# 
# - There is **NOT a meaningful relationship** between the independent variables **Alcoholism** and **Handcap** with our dependent variable Appointment status (No_Show)

# ### Question 6:  
# >### Top 10 neighborhoods with the highest number of no-show ups?

# In[151]:


# Top 10 neighbourhood with NO shows up
hoods_no_show=df.Neighbourhood[no_shows].value_counts().head(10)
hoods_no_show


# In[152]:


# Plot a Pie chart with Appointment status (showed up - not showed up)

labels = ['JARDIM CAMBURI','MARIA ORTIZ','ITARARÉ','RESISTÊNCIA',
          'CENTRO','JESUS DE NAZARETH','JARDIM DA PENHA','CARATOÍRA',
          'TABUAZEIRO','BONFIM']

colors = ['darkred', 'firebrick','indianred','palevioletred','lightcoral',
          'salmon','darksalmon','lightsalmon','darkorange','orange']

plt.pie(hoods_no_show, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
plt.title('\n\nTop 10 "No-Show Appointments" neighbourhoods\n\n',fontsize=18)
plt.axis('equal'); # Equal aspect ratio ensures that pie is drawn as a circle.


# ### Question 7:  
# >### Is there a specific day of the week in which more not show ups occur? 

# In[153]:


#Number of show ups and no show ups by weekdays
weekdays_vs_NoShow = pd.crosstab(index = df['Appointment_Weekday'], columns = df['No_Show'])
weekdays_vs_NoShow


# In[154]:


#Plot days vs Show ups and No show ups
days_shows = df['Appointment_Weekday'][shows].value_counts()
days_noshows= df['Appointment_Weekday'][no_shows].value_counts()

Appointment_Weekday=['Wednesday','Tuesday','Monday','Friday','Thursday','Saturday']

days_shows.plot(color='palegreen',linewidth=3.0,grid=.5)
days_noshows.plot(color='pink',linewidth=3.0,grid=.5)
plt.xlabel(Appointment_Weekday)
plt.ylabel('Number of Appointments')
plt.title('\nDays vs Appointment status\n')
L=plt.legend(fancybox=True, framealpha=0.9)
L.get_texts()[0].set_text('Showed up')
L.get_texts()[1].set_text('No Showed up');


# We can see from the graph how the curves of show and no-show appointments follow a very similar dynamic throughout the weekdays. **Wednesday is the day with the highest number of no show ups** (also with the highest number of show ups) while **saturday is the day with the lowest number of no show ups** (as well as for the show ups). However, we need to look at the data in proportion.

# In[155]:


#Create a table: Appointment_Weekday vs. No-show (to check proportionally)
weekdays_vs_no_show = pd.crosstab(index = df['Appointment_Weekday'], columns = df['No_Show'], normalize = 'index')
weekdays_vs_no_show


# In[156]:


# Plot a histogram chart Appointment_Weekday vs. Appointment status (showed up - not showed up)

colors = ['palegreen','pink']

weekdays_vs_no_show.plot(kind='bar', stacked=True, color=colors, figsize=(7,5))
plt.xlabel('Weekdays')
plt.ylabel('Number of Appointments')
plt.title('Appointment Weekday vs Appointment Status')
L=plt.legend(fancybox=True, framealpha=0.5)
L.get_texts()[0].set_text('Showed up')
L.get_texts()[1].set_text('No Showed up')


# From this graph we can conclude that the difference on Saturday might be due to the low number of records for this day, therefore, any Saturday  record represents a greater variation in proportion to the other days.

# ### Chi-Square Test
# #### Appointment Weekday vs No shows up

# In[157]:


#Conduct the Chi-Square Test(chi2), showing the p (p-value), dof(degrees of freedom) and ex(expected frequencies)
chi2, p, dof,ex = chi2_contingency(weekdays_vs_no_show, correction=False)
chi2, p


# The p-value is greater than 0.05% (0.99%), therefore, I am going not going to assume that **there is NOT a meaningful relationship between the appointment weekday and the appointment status.**

# ### Question 8:  
# >### Does the waiting days period of time affect the status of the appointment?

# In[158]:


df.columns


# In[159]:


#Create a contingency table: Appointment_Weekday vs. No-show (to check proportionally)
waiting_days_vs_no_show = pd.crosstab(index = df['Waiting_Days'], columns = df['No_Show'])
waiting_days_vs_no_show


# In[160]:


#Plot Waiting Days vs Show ups and No Show ups
waitingdays_shows = df['Waiting_Days'][shows].value_counts()
waitingdays_noshows= df['Waiting_Days'][no_shows].value_counts()

waitingdays_shows.plot(kind='bar', color='palegreen',linewidth=3.0, figsize=(20,10))
waitingdays_noshows.plot(kind='bar', color='pink',linewidth=3.0, figsize=(20,8))
plt.xlabel('Waiting Days',fontsize=20)
plt.ylabel('Number of Appointments',fontsize=20)
plt.title('\nWaiting Days vs Appointment Status\n',fontsize=24)
L=plt.legend(fancybox=True, framealpha=0.9, fontsize=20)
L.get_texts()[0].set_text('Showed up')
L.get_texts()[1].set_text('No Showed up');


# The most relevant insight of this plot is that a large majority scheduled the appointment the same days they attended, causing a very positive trend in attendance. **The shorten awaiting period the more patients show up.**

# <a id='conclusions'></a>
# ## Conclusions
# 
# 1. The majority of appointment status are **show up (79.8%)**, while those marked as **not show up** represent only the **20.2%** of the total number of records.
# 
# 
# 2. We can see in the chart above that **patients whose age is above 60 take appointment more seriously** than patients under that age.
# 
# 
# 3. Patients **between 9 and 45 years old** are those that represents a **greater proportion of no showed up appointmens,** while **patients above 60 take 
# appointment more seriously**
# 
# 
# 4. There are **more patients who did not attend the appointment and got a reminder SMS than those who did not attend it without having received it.**
# 
# 
# 5. The **Top 10 neighborhoods** with the highest number of **no-show ups** are, in order: Jardim Camburi, Maria Ortiz, Itararé, Resistência, Centro, Jesus De Nazareth, Jardim Da Penha, Caratoíra, Tabuazeiro Y Bonfim.
# 
# 
# 6. **Wednesdays and Tuesdays** are the days with the **highest number of appointments**, while **Thursday and Saturday** are the days with the **least number of appointments**, although there is no significant relationship between the date of the appointment and its final status.
# 
# 
# 7. **The shorten awaiting period the more patients show up.**
# 
# 
# >Variables that show a significant relationship with the status of the appointment according to statistical tests:
# >-	Age 
# >-	Scholarship
# >-	Hipertension
# >-	Diabetes
# >-	SMS_received
# 
# 
# 
# ## Limitations
# 
# • **The time period over the data is collected is relatively small** (6 months), if we had data referring to a longer period of time we would have drawn conclusions regarding seasonality and the rate of variation over the years.
# 
# • **The sample is very unbalanced** since we only have the 20% of records for non-show, which results in a strong limitation of the study of their causes.
# 
# • **Limitations of chi-square**: if the test result comes out as a meaningfult relathionship between the variables, that means there is some association but it does not provide further information. 
# 
# 

# In[163]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

