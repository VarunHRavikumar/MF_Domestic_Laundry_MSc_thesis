
"""

@author: Varun Harohalli Ravikumar
"""
# Importing libraries

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import f
import statsmodels.stats.api as sms
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from linearmodels.system import SUR


  ###################################################################################################
 #####################################  Reading the datasets ######################################
##################################################################################################

# Reading the data from the excel dataset for the large sample n=79

# Selecting the range of data to be analysed
range_of_cells = 'A:D'

# Reading data into python 
df = pd.read_excel('Working_dataset.xlsx', sheet_name='Consumer Wash Loads', header = [0], index_col = 0, usecols = range_of_cells )

# Convert index to string type
df.index = df.index.astype(str)

# Remove rows containing 'Mean' and 'Std Dev'
df = df[~df.index.str.contains('Mean|Standard Deviation')]

# Removing all the NaN values
df = df.dropna()


# Reading data into python 
range_of_cells = 'A:D'

# Reading the 40 degree wash data into python
temperature_40_df = pd.read_excel('Working_dataset.xlsx', sheet_name='Wash temperature', header = [1], index_col = 0, usecols = range_of_cells )

# Convert index to string type
temperature_40_df.index = temperature_40_df.index.astype(str)

# Remove rows containing 'Mean' and 'Std Dev'
temperature_40_df = temperature_40_df[~temperature_40_df.index.str.contains('Mean|Standard Deviation')]

# Removing all the NaN values
temperature_40_df = temperature_40_df.dropna()

MF_Release = temperature_40_df['Microfiber release (ppm)*']
Load_mass = temperature_40_df['Load mass (kg)']

# Reading the cold wash data into python 
range_of_cells = 'G:J'

# Reading data into python
temperature_cold_df = pd.read_excel('Working_dataset.xlsx', sheet_name='Wash temperature', header = [1], index_col = 0, usecols = range_of_cells )

# Convert index to string type
temperature_cold_df.index = temperature_cold_df.index.astype(str)

# Remove rows containing 'Mean' and 'Std Dev'
temperature_cold_df = temperature_cold_df[~temperature_cold_df.index.str.contains('Mean|Standard Deviation')]

# Removing all the NaN values
temperature_cold_df = temperature_cold_df.dropna()

MF_Release = temperature_cold_df['Microfiber release (ppm)*.1']
Load_mass = temperature_cold_df['Load mass (kg).1']



#%%

  ###################################################################################################
 ###################################  Statistical Analysis n=79 ####################################
##################################################################################################

# Group the data by 'Load mass (kg)' and calculate the average of 'Microfibre release (ppm)'
grouped_data = df.groupby('Load mass (kg)').mean()
load_mass_values = grouped_data.index
microfiber_release_values = grouped_data['Microfiber release (ppm)*']

# Scatter plot
plt.scatter(load_mass_values, microfiber_release_values, label='Actual Data')

# Define the logarithmic function
def logarithmic_func(x, a, b):
    return a * np.log(x) + b

# Fit the logarithmic function to the data
params, covariance = curve_fit(logarithmic_func, load_mass_values, microfiber_release_values)

# Extract the parameters a and b
a, b = params

# Generate the trendline data
trendline = a * np.log(load_mass_values) + b

# Plot the trendline
plt.plot(load_mass_values, trendline, color='red', label='Logarithmic Trendline')

plt.xlabel('Load mass (kg)')
plt.ylabel('Microfibre release (ppm)')
plt.title('Microfibre Release (ppm) data distribution with Logarithmic Trendline')
plt.legend()
plt.show()
plt.savefig('Data trend.png', dpi=1080, bbox_inches='tight')
#%%
# saving required data into variables

MF_Release = df['Microfiber release (ppm)*']
Load_mass = df['Load mass (kg)']

# Analysis of data distribution
# Shapiro-Wilk test

statistic, p_value = stats.shapiro(Load_mass)

print(statistic, p_value)

result = stats.anderson(Load_mass, dist='norm')
print("Anderson-Darling test:")
print("Statistic:", result.statistic)
print("Critical Values:", result.critical_values)
print("Significance Level:", result.significance_level)

result = stats.kstest(Load_mass, 'norm')

print("Kolmogorov-Smirnov test:")
print("Test Statistic:", result.statistic)
print("p-value:", result.pvalue)

if p_value < 0.05:
    
    # Converting data to normallise the distribution.
    log_data = np.log(df['Microfiber release (ppm)*'])
    df['Log-transformed'] = log_data
    selected_data = 'Log-transformed'
    print('\n','''Null hypothesis is not accepted as p < 0.05. 
 Therefore, the data is converted to normally distributed values using log transformation.''','\n')
else:
    
    # Selecting original data.
    log_data = df['Microfiber release (ppm)*']
    selected_data = 'Microfiber release (ppm)*'
    print('\n','''Null hypothesis is accepted as p > 0.05. 
          Therefore, the data is normally distributed''','\n')
    
#%%

# Calculating mean
mean = np.mean(log_data)
print('Mean:', round(mean,3))

# Calculating the standard error of mean
sem_value = stats.sem(log_data)
print('Standard error of Mean:', round(sem_value,3))

# Calculating the confidence interval
CI = sms.DescrStatsW(log_data).tconfint_mean()
print("95% Confidence Interval for Mean:", (round(CI[0], 2), round(CI[1], 2)))

# Calculating median
median = np.median(log_data)
print('Median:', round(median,3))

# Calculating variance
variance  = np.var(log_data)
print('Variance:', round(variance,3))

# Calculating standard deviation
std_dev = np.std(log_data)
print('Standard Deviation:', round(std_dev,3))

# Calculating maximum
maximum = np.max(log_data)
print('Maximum:', round(maximum,3))

# Calculating minimum
minimum = np.min(log_data)
print('Minimum:', round(minimum,3))

# Calculating range
data_range = np.ptp(log_data)
print('Range:', round(data_range,3))

# Calculating interquartile range
Q1 = np.percentile(log_data, 25)
Q3 = np.percentile(log_data, 75)
IQR = Q3 - Q1
print('Interquartile Range:', round(IQR,3), '\n')

# Performing skew test
skewness_test, skewness_p_val = stats.skewtest(log_data)
print("Skewness test:")
print("Test statistic:", round(skewness_test,3))
print("p-value:", round(skewness_p_val,3), '\n')

# Performing kurtosis test
kurtosis_test, kurtosis_p_val = stats.kurtosistest(log_data)
print("Kurtosis test:")
print("Test statistic:", round(kurtosis_test,3))
print("p-value:", round(kurtosis_p_val,3),'\n')


#%%

# Checking for outliers

# Defining the threshold for the Z-score calculation
z_score_threshold = 3

# Calculating the Z-scores
z_score = np.abs((log_data - log_data.mean()) / log_data.std())

# Identifying the outlying values based on Z-scores exceeding the threshold in both positive and negative directions
outliers = df[(z_score > z_score_threshold) | (z_score < -z_score_threshold)][selected_data]

if outliers.empty:
    print('No outliers detected.','\n')
else:
    print('Outlier values:', round(outliers, 3),'\n')


#%%

# Calculating the degree of freedom
n = len(log_data)
deg_f = n - 2

# Testing for Normality
# Shapiro-Wilk test
statistic, p_value = stats.shapiro(log_data)

# Printing the results
print("Shapiro-Wilk test:")
print("Statistic:", round(statistic,3))
print("p-value:", round(p_value,3))
print('Degree of freedom:', deg_f,'\n')

result = stats.anderson(log_data, dist='norm')
print("Anderson-Darling test:")
print("Statistic:", result.statistic)
print("Critical Values:", result.critical_values)
print("Significance Level:", result.significance_level)
#%%

# Plotting the distribution curve

plt.figure(figsize=(8, 6))
plt.hist(log_data, bins=30, density=True, alpha=0.5)


# Fitting a probability distribution function to the data
mu, sigma = stats.norm.fit(log_data)
x = np.linspace(log_data.min(), log_data.max(), 100)
pdf = stats.norm.pdf(x, mu, sigma)

# Plotting the curve over the histogram
plt.plot(x, pdf, 'r-', label='Normal Distribution')

# Setting labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution Curve of ' + 'Microfiber release (ppm)*')
plt.legend()
plt.show()
plt.savefig('Distribution curve n79.png', dpi=1080, bbox_inches='tight')

#%%

# Generating the Q-Q plot
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(log_data , dist="norm", plot=ax)
ax.set_title('Q-Q Plot for ' + 'Microfiber release (ppm)*')
plt.show()
plt.savefig('Q-Q plot n79.png', dpi=1080, bbox_inches='tight')
#%%

# Box and whisker plot for original Load mass v Microfibre release 

value_set = ['Load mass (kg)', 'Microfiber release (ppm)*']
data_to_plot = [df['Load mass (kg)'], df['Microfiber release (ppm)*']]

# Generating the box and whisker plot
fig, ax = plt.subplots(figsize=(8, 6))


# Positions for the box plots
pos = [2, 1]


# Plotting the first y-axis (Microfiber release)
ax.boxplot(data_to_plot[1], positions=[pos[1]], labels=[value_set[1]])
ax.set_ylabel(value_set[1])


# Creating a second y-axis for Load mass
ax2 = ax.twinx()
ax2.boxplot(data_to_plot[0], positions=[pos[0]], labels=[value_set[0]])
ax2.set_ylabel(value_set[0])


ax.set_title('Box and Whisker Plot')
plt.show()
plt.savefig('Box plot n79.png', dpi=1080, bbox_inches='tight')
#%%

import statsmodels.api as sm

# Creating the regression model
X0 = np.log(df['Load mass (kg)'])
y0 = np.log(df['Microfiber release (ppm)*'])
X0 = sm.add_constant(X0)  # Adding a constant term to the predictor variable

model = sm.OLS(y0, X0)  # Creating an ordinary least squares (OLS) model
results = model.fit()  # Fitting the model to the data

# Printing the regression results
print(results.summary())

print('\n')
# Predicting the release of MFs
X_pred = np.log(df['Load mass (kg)'])  # Load mass to predict the MFs release values
X_pred = sm.add_constant(X_pred)  # Adding a constant term to the predictor variable

y_pred = results.predict(X_pred)  # Predicting the release of MFs

# Printing the regression equation
intercept = results.params[0]
slope = results.params[1]
equation = f"y = {intercept:.3f} + {slope:.3f} * X"

 
print("Regression Equation:")
print(equation)


#%%
  ###################################################################################################
 ####################################  Statistical Analysis 40 #####################################
##################################################################################################
# Analysis of data distribution
# Shapiro-Wilk test

statistic, p_value = stats.shapiro(MF_Release)

if p_value < 0.05:
    
    # Converting data to normallise the distribution.
    log_data_temp40 = np.log(temperature_40_df['Microfiber release (ppm)*'])
    temperature_40_df['Log-transformed'] = log_data_temp40
    selected_data_temp40 = 'Log-transformed'
    print('\n','''Null hypothesis is not accepted as p < 0.05. 
 Therefore, the data is converted to normally distributed values using log transformation.''','\n')
else:
    
    # Selecting original data.
    log_data_temp40 = temperature_40_df['Microfiber release (ppm)*']
    selected_data_temp40 = 'Microfiber release (ppm)*'
    print('\n','''Null hypothesis is accepted as p > 0.05. 
          Therefore, the data is normally distributed''','\n')
    
#%%

# Calculating mean
mean = np.mean(log_data_temp40)
print('Mean:', round(mean,3))

# Calculating the standard error of mean
sem_value = stats.sem(log_data_temp40)
print('Standard error of Mean:', round(sem_value,3))

# Calculating the confidence interval
CI = sms.DescrStatsW(log_data_temp40).tconfint_mean()
print("95% Confidence Interval for Mean:", (round(CI[0], 2), round(CI[1], 2)))

# Calculating median
median = np.median(log_data_temp40)
print('Median:', round(median,3))

# Calculating variance
variance  = np.var(log_data_temp40)
print('Variance:', round(variance,3))

# Calculating standard deviation
std_dev = np.std(log_data_temp40)
print('Standard Deviation:', round(std_dev,3))

# Calculating maximum
maximum = np.max(log_data_temp40)
print('Maximum:', round(maximum,3))

# Calculating minimum
minimum = np.min(log_data_temp40)
print('Minimum:', round(minimum,3))

# Calculating range
data_range = np.ptp(log_data_temp40)
print('Range:', round(data_range,3))

# Calculating interquartile range
Q1 = np.percentile(log_data_temp40, 25)
Q3 = np.percentile(log_data_temp40, 75)
IQR = Q3 - Q1
print('Interquartile Range:', round(IQR,3), '\n')

# Performing skew test
skewness_test, skewness_p_val = stats.skewtest(log_data_temp40)
print("Skewness test:")
print("Test statistic:", round(skewness_test,3))
print("p-value:", round(skewness_p_val,3), '\n')

# Performing kurtosis test
kurtosis_test, kurtosis_p_val = stats.kurtosistest(log_data_temp40)
print("Kurtosis test:")
print("Test statistic:", round(kurtosis_test,3))
print("p-value:", round(kurtosis_p_val,3),'\n')


#%%

# Checking for outliers

# Defining the threshold for the Z-score calculation
z_score_threshold = 3

# Calculating the Z-scores
z_score = np.abs((log_data_temp40 - log_data_temp40.mean()) / log_data_temp40.std())

# Identifying the outlying values based on Z-scores exceeding the threshold in both positive and negative directions
outliers = temperature_40_df[(z_score > z_score_threshold) | (z_score < -z_score_threshold)][selected_data_temp40]

if outliers.empty:
    print('No outliers detected.','\n')
else:
    print('Outlier values:', round(outliers, 3),'\n')


#%%

# Calculating the degree of freedom
n = len(log_data_temp40)
deg_f = n - 2

# Testing for Normality
# Shapiro-Wilk test
statistic, p_value = stats.shapiro(log_data_temp40)

# Printing the results
print("Shapiro-Wilk test:")
print("Statistic:", round(statistic,3))
print("p-value:", round(p_value,3))
print('Degree of freedom:', deg_f,'\n')

# Testing for Normality
# Shapiro-Wilk test
statistic, p_value = stats.shapiro(Load_mass)
if p_value < 0.05:
    print('null rejected')
else:
    print('null accepted')
    
# Printing the results
print("Shapiro-Wilk test:")
print("Statistic:", round(statistic,3))
print("p-value:", round(p_value,3))
print('Degree of freedom:', deg_f,'\n')
#%%

# Plotting the distribution curve

plt.figure(figsize=(8, 6))
plt.hist(log_data_temp40, bins=30, density=True, alpha=0.5)


# Fitting a probability distribution function to the data
mu, sigma = stats.norm.fit(log_data_temp40)
x = np.linspace(log_data_temp40.min(), log_data_temp40.max(), 100)
pdf = stats.norm.pdf(x, mu, sigma)

# Plotting the curve over the histogram
plt.plot(x, pdf, 'r-', label='Normal Distribution')

# Setting labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution Curve of ' + 'Microfiber release (ppm)*')
plt.legend()
plt.show()
plt.savefig('Distribution curve 40degree.png', dpi=1080, bbox_inches='tight')

#%%

# Generating the Q-Q plot
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(log_data_temp40 , dist="norm", plot=ax)
ax.set_title('Q-Q Plot for ' + 'Microfiber release (ppm)*')
plt.show()

#%%

# Generating the box and whisker plot for 'Microfiber release (ppm)*'
plt.figure(figsize=(8, 6))
plt.boxplot(log_data_temp40)

# Set labels and title
plt.xlabel('Column')
plt.ylabel('Values')
plt.title('Box and Whisker Plot - Microfiber release (ppm)*')
plt.show()
plt.savefig('Q-Q plot 40degree.png', dpi=1080, bbox_inches='tight')

# Box and whisker plot for original Load mass v Microfibre release
value_set = ['Load mass (kg)', 'Microfiber release (ppm)*']
data_to_plot = [temperature_40_df['Load mass (kg)'], temperature_40_df['Microfiber release (ppm)*']]

# Generating the box and whisker plot
fig, ax = plt.subplots(figsize=(8, 6))

# Positions for the box plots
pos = [2, 1]

# Plotting the first y-axis (Microfiber release)
ax.boxplot(data_to_plot[1], positions=[pos[1]], labels=[value_set[1]])
ax.set_ylabel(value_set[1])

# Creating a second y-axis for Load mass
ax2 = ax.twinx()
ax2.boxplot(data_to_plot[0], positions=[pos[0]], labels=[value_set[0]])
ax2.set_ylabel(value_set[0])

ax.set_title('Box and Whisker Plot')

plt.show()
plt.savefig('Box plot 40degree.png', dpi=1080, bbox_inches='tight')

#%%
#Regression model 40 degree wash

import statsmodels.api as sm

# Creating the regression model
X1 = np.log(temperature_40_df['Load mass (kg)'])
y1 = log_data_temp40
X1 = sm.add_constant(X1)  # Adding a constant term to the predictor variable

model = sm.OLS(y1, X1)  # Creating an ordinary least squares (OLS) model
results = model.fit()  # Fitting the model to the data

# Printing the regression results
print(results.summary())

print('\n')
# Predicting the release of MFs
X_pred1 = np.log(temperature_40_df['Load mass (kg)'])  # Load mass values for prediction
X_pred1 = sm.add_constant(X_pred1)  # Adding a constant term to the predictor variable

y_pred1 = results.predict(X_pred1)  # Predicting the release of MFs

# Printing the regression equation
intercept1 = results.params[0]
slope1 = results.params[1]
equation1 = f"y = {intercept1:.3f} + {slope1:.3f} * X"
print("Regression Equation:")
print(equation1)

#%%
#Sensitivity analysis 40 degree wash

# Creating the regression model
X1 = np.log(temperature_40_df['Load mass (kg)'])
y1 = log_data_temp40
X1 = sm.add_constant(X1)  # Adding a constant term to the predictor variable

model = sm.OLS(y1, X1)  # Creating an ordinary least squares (OLS) model
results = model.fit()  # Fitting the model to the data

# Printing the regression results
print(results.summary())
print('\n')

# Sensitivity Analysis
min_X1 = X1['Load mass (kg)'].min()  # Minimum value of the independent variable
max_X1 = X1['Load mass (kg)'].max()  # Maximum value of the independent variable
num_points = 100  # Number of points in the range

X_pred1 = np.linspace(min_X1, max_X1, num_points)  # Range of values for the independent variable
X_pred1 = sm.add_constant(X_pred1)  # Adding a constant term to the predictor variable

y_pred1 = results.predict(X_pred1)  # Predicting the dependent variable

# Printing the regression equation
intercept1 = results.params[0]
slope1 = results.params[1]
equation1 = f"y = {intercept1:.3f} + {slope1:.3f} * X"
print("Regression Equation:")
print(equation1)

# Sensitivity Analysis Results
print("\nSensitivity Analysis:")
for i, x_val in enumerate(X_pred1[:, 1]):
    print(f"X = {x_val:.3f}, Predicted y = {y_pred1[i]:.3f}")

# Plotting the sensitivity analysis results
plt.scatter(temperature_40_df['Load mass (kg)'], np.exp(log_data_temp40), label='Actual Data')
plt.scatter(np.exp(X_pred1[:, 1]), np.exp(y_pred1), color='red', marker ='2', label='Predicted Data')
plt.xlabel('Load mass (kg)')
plt.ylabel('Microfibre release (ppm)')
plt.title('Sensitivity Analysis (40 degrees wash)')
plt.legend()
plt.show()
plt.savefig('Sensitivity 40degree.png', dpi=1080, bbox_inches='tight')
#%%

# Performing a residual analysis
residuals = results.resid

# Plotting the residuals
plt.figure(figsize=(8, 6))
plt.scatter(np.exp(X1['Load mass (kg)']), residuals)
plt.axhline(y=0, color='red', linestyle='--')  # Adding a horizontal line at y=0
plt.xlabel('Load mass (kg)')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()
plt.savefig('residual plot 40degree.png', dpi=1080, bbox_inches='tight')

# Checking the distribution of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.5)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
plt.savefig('Residuals distribution plot 40degree.png', dpi=1080, bbox_inches='tight')
#%%

# Obtain the predicted values
predicted = results.predict(X1)

# Plotting residuals against the predicted values
plt.figure(figsize=(8, 6))
plt.scatter(predicted, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted values')
plt.show()

# Performing the Breusch-Pagan test
import statsmodels.stats.diagnostic as sm_diag

_, p_value, _, _ = sm_diag.het_breuschpagan(residuals, X1)
alpha = 0.05  # significance level

if p_value < alpha:
    print("The data exhibits heteroscedasticity (reject null hypothesis).")
else:
    print("The data does not exhibit heteroscedasticity (fail to reject null hypothesis).")

# Performing the White test
_, p_value, _, _ = sm_diag.het_white(residuals, X1)

if p_value < alpha:
    print("The data exhibits heteroscedasticity (reject null hypothesis).")
else:
    print("The data does not exhibit heteroscedasticity (fail to reject null hypothesis).")

#%%
  ###################################################################################################
 ###################################  Sensitivity Analysis cold ####################################
##################################################################################################

# Analysis of data distribution
# Shapiro-Wilk test

statistic, p_value = stats.shapiro(MF_Release)

                      #######################################################################################
# if p_value < 0.05: ## This has been commented out due to the distribution of cold wash data being normal ##
                     #######################################################################################

#     # Converting data to normallise the distribution.
#     log_data_cold = np.log(temperature_cold_df['Microfiber release (ppm)*.1'])
#     temperature_cold_df['Log-transformed'] = log_data_cold
#     selected_data = 'Log-transformed'
#     print('\n','''Null hypothesis is not accepted as p < 0.05. 
#  Therefore, the data is converted to normally distributed values using log transformation.''','\n')
# else:
    
#     # Selecting original data.
#     log_data_cold = temperature_cold_df['Microfiber release (ppm)*.1']
#     selected_data = 'Microfiber release (ppm)*.1'
#     print('\n','''Null hypothesis is accepted as p > 0.05. 
#           Therefore, the data is normally distributed''','\n')
    
# log transforming the the distribution.
log_data_cold = np.log(temperature_cold_df['Microfiber release (ppm)*.1'])
temperature_cold_df['Log-transformed'] = log_data_cold
selected_data = 'Log-transformed'
print('\n','''Null hypothesis is not accepted as p < 0.05. 
  Therefore, the data is converted to normally distributed values using log transformation.''','\n')
#%%

# Calculating mean
mean = np.mean(log_data_cold)
print('Mean:', round(mean,3))

# Calculating the standard error of mean
sem_value = stats.sem(log_data_cold)
print('Standard error of Mean:', round(sem_value,3))

# Calculating the confidence interval
CI = sms.DescrStatsW(log_data_cold).tconfint_mean()
print("95% Confidence Interval for Mean:", (round(CI[0], 2), round(CI[1], 2)))

# Calculating median
median = np.median(log_data_cold)
print('Median:', round(median,3))

# Calculating variance
variance  = np.var(log_data_cold)
print('Variance:', round(variance,3))

# Calculating standard deviation
std_dev = np.std(log_data_cold)
print('Standard Deviation:', round(std_dev,3))

# Calculating maximum
maximum = np.max(log_data_cold)
print('Maximum:', round(maximum,3))

# Calculating minimum
minimum = np.min(log_data_cold)
print('Minimum:', round(minimum,3))

# Calculating range
data_range = np.ptp(log_data_cold)
print('Range:', round(data_range,3))

# Calculating interquartile range
Q1 = np.percentile(log_data_cold, 25)
Q3 = np.percentile(log_data_cold, 75)
IQR = Q3 - Q1
print('Interquartile Range:', round(IQR,3), '\n')

# Performing skew test
skewness_test, skewness_p_val = stats.skewtest(log_data_cold)
print("Skewness test:")
print("Test statistic:", round(skewness_test,3))
print("p-value:", round(skewness_p_val,3), '\n')

# Performing kurtosis test
kurtosis_test, kurtosis_p_val = stats.kurtosistest(log_data_cold)
print("Kurtosis test:")
print("Test statistic:", round(kurtosis_test,3))
print("p-value:", round(kurtosis_p_val,3),'\n')


#%%

# Checking for outliers

# Defining the threshold for the Z-score calculation
z_score_threshold = 3

# Calculating the Z-scores
z_score = np.abs((log_data_cold - log_data_cold.mean()) / log_data_cold.std())

# Identifying the outlying values based on Z-scores exceeding the threshold in both positive and negative directions
outliers = temperature_cold_df[(z_score > z_score_threshold) | (z_score < -z_score_threshold)][selected_data]

if outliers.empty:
    print('No outliers detected.','\n')
else:
    print('Outlier values:', round(outliers, 3),'\n')


#%%

# Calculating the degree of freedom
n = len(log_data_cold)
deg_f = n - 2

# Testing for Normality
# Shapiro-Wilk test
statistic, p_value = stats.shapiro(log_data_cold)

# Printing the results
print("Shapiro-Wilk test:")
print("Statistic:", round(statistic,3))
print("p-value:", round(p_value,3))
print('Degree of freedom:', deg_f,'\n')

# Testing for Normality
# Shapiro-Wilk test
statistic, p_value = stats.shapiro(Load_mass)
if p_value < 0.05:
    print('null rejected')
else:
    print('null accepted')
    
# Printing the results
print("Shapiro-Wilk test:")
print("Statistic:", round(statistic,3))
print("p-value:", round(p_value,3))
print('Degree of freedom:', deg_f,'\n')

#%%

# Plotting the distribution curve

plt.figure(figsize=(8, 6))
plt.hist(log_data_cold, bins=30, density=True, alpha=0.5)


# Fitting a probability distribution function to the data
mu, sigma = stats.norm.fit(log_data_cold)
x = np.linspace(log_data_cold.min(), log_data_cold.max(), 100)
pdf = stats.norm.pdf(x, mu, sigma)

# Plotting the curve over the histogram
plt.plot(x, pdf, 'r-', label='Normal Distribution')

# Setting labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution Curve of ' + 'Microfiber release (ppm)*')
plt.legend()
plt.show()
plt.savefig('Distribution curve cold.png', dpi=1080, bbox_inches='tight')
#%%

# Generating the Q-Q plot
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(log_data_cold , dist="norm", plot=ax)
ax.set_title('Q-Q Plot for ' + 'Microfiber release (ppm)*')
plt.show()
plt.savefig('Q-Q plot cold.png', dpi=1080, bbox_inches='tight')
#%%

# Box and whisker plot for original Load mass v Microfibre release
value_set = ['Load mass (kg)', 'Microfiber release (ppm)*']
data_to_plot = [temperature_cold_df['Load mass (kg).1'], temperature_cold_df['Microfiber release (ppm)*.1']]

# Generating the box and whisker plot
fig, ax = plt.subplots(figsize=(8, 6))

# Positions for the box plots
pos = [2, 1]

# Plotting the first y-axis (Microfiber release)
ax.boxplot(data_to_plot[1], positions=[pos[1]], labels=[value_set[1]])
ax.set_ylabel(value_set[1])

# Creating a second y-axis for Load mass
ax2 = ax.twinx()
ax2.boxplot(data_to_plot[0], positions=[pos[0]], labels=[value_set[0]])
ax2.set_ylabel(value_set[0])

ax.set_title('Box and Whisker Plot')

plt.show()
plt.savefig('Box plot cold.png', dpi=1080, bbox_inches='tight')

#%%
#regression model cold wash

import statsmodels.api as sm

# Creating the regression model
X2 = np.log(temperature_cold_df['Load mass (kg).1'])
y2 = log_data_cold
X2 = sm.add_constant(X2)  # Adding a constant term to the predictor variable

model = sm.OLS(y2, X2)  # Creating an ordinary least squares (OLS) model
results = model.fit()  # Fitting the model to the data

# Printing the regression results
print(results.summary())

print('\n')
# Predicting the dependent variable
X_pred2 = np.log(temperature_cold_df['Load mass (kg).1'])  # Independent variable for prediction
X_pred2 = sm.add_constant(X_pred2)  # Adding a constant term to the predictor variable

y_pred2 = results.predict(X_pred2)  # Predicting the dependent variable

# Printing the regression equation
intercept2 = results.params[0]
slope2 = results.params[1]
equation2 = f"y = {intercept2:.3f} + {slope2:.3f} * X"
print("Regression Equation:")
print(equation2)
print('\n')

#%%
# sensitivity analysis cold wash
# Creating the regression model
X2 = np.log(temperature_cold_df['Load mass (kg).1'])
y2 = log_data_cold
X2 = sm.add_constant(X2)  # Adding a constant term to the predictor variable

model = sm.OLS(y2, X2)  # Creating an ordinary least squares (OLS) model
results = model.fit()  # Fitting the model to the data

# Printing the regression results
print(results.summary())
print('\n')

# Sensitivity Analysis
min_X2 = X2['Load mass (kg).1'].min()  # Minimum value of the independent variable
max_X2 = X2['Load mass (kg).1'].max()  # Maximum value of the independent variable
num_points = 100  # Number of points in the range

X_pred2 = np.linspace(min_X2, max_X2, num_points)  # Range of values for the independent variable
X_pred2 = sm.add_constant(X_pred2)  # Adding a constant term to the predictor variable

y_pred2 = results.predict(X_pred2)  # Predicting the dependent variable

# Printing the regression equation
intercept2 = results.params[0]
slope2 = results.params[1]
equation2 = f"y = {intercept2:.3f} + {slope2:.3f} * X"
print("Regression Equation:")
print(equation2)

# Sensitivity Analysis Results
print("\nSensitivity Analysis:")
for i, x_val in enumerate(X_pred2[:, 1]):
    print(f"X = {x_val:.3f}, Predicted y = {y_pred2[i]:.3f}")

# Plotting the sensitivity analysis results
plt.scatter(temperature_cold_df['Load mass (kg).1'], np.exp(log_data_cold), label='Actual Data')
plt.scatter(np.exp(X_pred2[:, 1]), np.exp(y_pred2), color='red', marker ='2', label='Predicted Data', s=50)
plt.xlabel('Load mass (kg)')
plt.ylabel('Microfibre release (ppm)')
plt.title('Sensitivity Analysis (Cold wash)')
plt.legend()
plt.show()
plt.savefig('Sensitivity cold.png', dpi=1080, bbox_inches='tight')
#%%

# Performing a residual analysis
residuals = results.resid

# Plotting the residuals
plt.figure(figsize=(8, 6))
plt.scatter(np.exp(X2['Load mass (kg).1']), residuals)
plt.axhline(y=0, color='red', linestyle='--')  # Adding a horizontal line at y=0
plt.xlabel('Load mass (kg)')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()
plt.savefig('Residual plot cold.png', dpi=1080, bbox_inches='tight')
# Checking the distribution of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.5)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
plt.savefig('Residual distribution plot cold.png', dpi=1080, bbox_inches='tight')
#%%

# Obtain the predicted values
predicted = results.predict(X1)

# Plotting residuals against the predicted values
plt.figure(figsize=(8, 6))
plt.scatter(predicted, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted values')
plt.show()

# Performing the Breusch-Pagan test
import statsmodels.stats.diagnostic as sm_diag

_, p_value, _, _ = sm_diag.het_breuschpagan(residuals, X1)
alpha = 0.05  # significance level

if p_value < alpha:
    print("The data exhibits heteroscedasticity (reject null hypothesis).")
else:
    print("The data does not exhibit heteroscedasticity (fail to reject null hypothesis).")

# Performing the White test
_, p_value, _, _ = sm_diag.het_white(residuals, X2)

if p_value < alpha:
    print("The data exhibits heteroscedasticity (reject null hypothesis).")
else:
    print("The data does not exhibit heteroscedasticity (fail to reject null hypothesis).")


#%%
  ###################################################################################################
 ############################################  T-test #############################################
##################################################################################################
# Perform t-test to analyse if there exist a significant difference between the means of the two data
t_statistic, p_value = ttest_ind(log_data_temp40, log_data_cold)

# Print the results
print("T-Test results:")
print("Test statistic:", round(t_statistic, 3))
print("p-value:", round(p_value, 3))

if p_value < 0.05:
    print("There is a significant difference between the means.")
else:
    print("There is no significant difference between the means.")

# Data for No Detergent
cycle1_no_detergent = [111.55, 70.24, 52.7, 45.02]
cycle4_no_detergent = [41.22, 39.12, 40.6, 33.69]
cycle8_no_detergent = [28.22, 36.72, 15.68, 17.39]

# Data for European Pod
cycle1_detergent = [66.26, 70.31, 83.59, 61.28]
cycle4_detergent = [52.34, 54.77, 31.47, 53.19]
cycle8_detergent = [32.82, 38.61, 18.82, 21.42]

# Perform t-test for each cycle
def perform_t_test(sample1, sample2):
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    return t_stat, p_value

def analyze_data(cycle_no, sample1, sample2):
    print(f"Cycle {cycle_no}:")
    t_stat, p_value = perform_t_test(sample1, sample2)
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  There is a significant impact of detergents on microfiber release.")
    else:
        print("  There is no significant impact of detergents on microfiber release.")

# Perform t-tests for each cycle
analyze_data(1, cycle1_detergent, cycle1_no_detergent)
analyze_data(4, cycle4_detergent, cycle4_no_detergent)
analyze_data(8, cycle8_detergent, cycle8_no_detergent)

# Data for "European Pod"
european_pod_microfiber = [
    [96.1, 70.74, 76.41, 71.68],  # Cycle 1
    [53.52, 40.92, 55.82, 29.61],  # Cycle 4
    [31.58, 21.71, 21.5, 29.1],    # Cycle 8
    [28.24, 33.6],                 # Cycle 16 (Note: Only 2 data points)
    [27.91, 41.07],                # Cycle 32 (Note: Only 2 data points)
    [17.94, 19.47]                 # Cycle 48 (Note: Only 2 data points)
]

# Data for "European Pod + Fabric Softener"
european_pod_fabric_microfiber = [
    [83.31, 87.04, 69.91, 90.63],  # Cycle 1
    [26.99, 75.31, 47.45, 85.97],  # Cycle 4
    [17, 66.58, 24.93, 37.62],     # Cycle 8
    [30.44, 24.55],                # Cycle 16 (Note: Only 2 data points)
    [22.66, 23.87],                # Cycle 32 (Note: Only 2 data points)
    [25.06, 29.46]                 # Cycle 48 (Note: Only 2 data points)
]

# Define the significance level
alpha = 0.05

for cycle in range(len(european_pod_microfiber)):
    print(f"Cycle {cycle+1}:")

    # Perform the two-sample t-test
    t_statistic, p_value = stats.ttest_ind(
        european_pod_microfiber[cycle], european_pod_fabric_microfiber[cycle]
    )

    # Compare the p-value with the significance level
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in microfiber release.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in microfiber release.")

    print()  # Add a new line for better readability


#%%
  ###################################################################################################
 #############################################  SUR  ##############################################
##################################################################################################
# equation1 = "y = {intercept1:.3f} + {slope1:.3f} * X"

# equation2 = "y = {intercept2:.3f} + {slope2:.3f} * X"

# equation1 = "y1 ~ np.log(temperature_40_df['Load mass (kg)'])"
# equation2 = "y2 ~ np.log(temperature_cold_df['Load mass (kg).1'])"

equation1 = "y1 ~ X1"
equation2 = "y2 ~ X2"

#formulas = [equation1, equation2]

#temperature_cold_df0 = temperature_cold_df.rename(columns = {'Load mass (kg).1':'Load mass (kg)', 'Microfiber mass (mg).1':'Microfiber mass (mg)', 'Microfiber release (ppm)*.1':'Microfiber release (ppm)*'})

combined_df = pd.concat([temperature_40_df, temperature_cold_df],
    axis=1,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    copy=True,)

# Estimate the SUR model

formulas = {'equation1': equation1, 'equation2': equation2}
sur_model = SUR.from_formula(formulas, combined_df)
sur_results = sur_model.fit(cov_type='robust')

# View the regression results
print(sur_results.summary)

cov_matrix = sur_results.cov

# Define the null hypothesis matrix
null_hypothesis = 'X1 = X2'

# Extract the coefficient estimates
coef_estimates = sur_results.params

# Extract the covariance matrix
cov_matrix = sur_results.cov

# Construct the Wald test
R = np.eye(len(coef_estimates))
R[0, 1] = -1  # Assuming X1 corresponds to the first equation and X2 corresponds to the second equation
wald_statistic = (R @ coef_estimates).T @ np.linalg.inv(R @ cov_matrix @ R.T) @ (R @ coef_estimates)
wald_pvalue = 1 - f.cdf(wald_statistic, len(R), len(sur_results.resids))
wald_pvalue1 = 1 - stats.chi2.cdf(wald_statistic, len(R))
# Print the results
print('\n')
print("Wald test statistic:", wald_statistic.item())
print("Wald test p-value:", wald_pvalue.item())
print("Wald test p-value:", wald_pvalue1.item())


#%%

import numpy as np
import matplotlib.pyplot as plt

# Generate random values between 1 and 8
random = np.random.uniform(1, 8, size=100)
random_values = np.log(random)
# Creating the regression model
X1 = np.log(temperature_40_df['Load mass (kg)'])
y1 = log_data_temp40
X1 = sm.add_constant(X1)  # Adding a constant term to the predictor variable

model = sm.OLS(y1, X1)  # Creating an ordinary least squares (OLS) model
results = model.fit()  # Fitting the model to the data

# Printing the regression results
print(results.summary())
print('\n')

# Sensitivity Analysis
min_X1 = random_values.min()  # Minimum value of the Load mass value
max_X1 = random_values.max()  # Maximum value of the Load mass value
num_points = 100  # Number of points in the range

X_pred1 = np.linspace(min_X1, max_X1, num_points)  # Range of values for the Load mass values
X_pred1 = sm.add_constant(X_pred1)  # Adding a constant term to the predictor variable

y_pred1 = results.predict(X_pred1)  # Predicting the release of MFs

# Printing the regression equation
intercept1 = results.params[0]
slope1 = results.params[1]
equation1 = f"y = {intercept1:.3f} + {slope1:.3f} * X"
print("Regression Equation:")
print(equation1)

# Sensitivity Analysis Results
print("\nSensitivity Analysis:")
for i, x_val in enumerate(random_values):
    print(f"X = {x_val:.3f}, Predicted y = {y_pred1[i]:.3f}")
    
# Plot the data
plt.scatter(np.exp(X_pred1[:, 1]), np.exp(y_pred1), color='red', label='Predicted Data', s= 5)
plt.xlabel('Load mass (kg)')
plt.ylabel('Microfibre release (ppm)')
plt.title('Sensitivity Analysis (40 degree wash)')
plt.legend()
plt.show()
plt.savefig('Sensitivity 40 predicted.png', dpi=1080, bbox_inches='tight')

plt.scatter(temperature_40_df['Load mass (kg)'], np.exp(log_data_temp40), label='Actual Data')
plt.xlabel('Load mass (kg)')
plt.ylabel('Microfibre release (ppm)')
plt.title('Sensitivity Analysis (40 degree wash)')
plt.legend()
plt.show()
plt.savefig('Sensitivity 40 Actual.png', dpi=1080, bbox_inches='tight')

# Creating the regression model for cold temperature
X1 = np.log(temperature_cold_df['Load mass (kg).1'])
y1 = log_data_cold
X1 = sm.add_constant(X1)  # Adding a constant term to the predictor variable

model = sm.OLS(y1, X1)  # Creating an ordinary least squares (OLS) model
results = model.fit()  # Fitting the model to the data

# Printing the regression results
print(results.summary())
print('\n')

# Sensitivity Analysis
min_X1 = random_values.min()  # Minimum value of the independent variable
max_X1 = random_values.max()  # Maximum value of the independent variable
num_points = 100  # Number of points in the range

X_pred1 = np.linspace(min_X1, max_X1, num_points)  # Range of values for the independent variable
X_pred1 = sm.add_constant(X_pred1)  # Adding a constant term to the predictor variable

y_pred1 = results.predict(X_pred1)  # Predicting the dependent variable

# Printing the regression equation
intercept1 = results.params[0]
slope1 = results.params[1]
equation1 = f"y = {intercept1:.3f} + {slope1:.3f} * X"
print("Regression Equation:")
print(equation1)

# Sensitivity Analysis Results
print("\nSensitivity Analysis:")
for i, x_val in enumerate(random_values):
    print(f"X = {x_val:.3f}, Predicted y = {y_pred1[i]:.3f}")
    
# Plot the data
plt.scatter(np.exp(X_pred1[:, 1]), np.exp(y_pred1), color='red', label='Predicted Data', s= 5)
plt.xlabel('Load mass (kg)')
plt.ylabel('Microfibre release (ppm)')
plt.title('Sensitivity Analysis (Cold wash)')
plt.legend()
plt.show()
plt.savefig('Sensitivity cold Predicted.png', dpi=1080, bbox_inches='tight')

plt.scatter(temperature_cold_df['Load mass (kg).1'], np.exp(log_data_cold), label='Actual Data')
plt.xlabel('Load mass (kg)')
plt.ylabel('Microfibre release (ppm)')
plt.title('Sensitivity Analysis (Cold wash)')
plt.legend()
plt.show()
plt.savefig('Sensitivity cold Actual.png', dpi=1080, bbox_inches='tight')
