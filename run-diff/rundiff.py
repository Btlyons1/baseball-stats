# Import libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# %matplotlib inline

fp = "gl2010_17/"

# Run differential is a cumulative team statistic that combines offensive and defensive scoring.
# Run differential is calculated by subtracting runs allowed from runs scored.
# The run differential is positive if a team scores more runs than it allows, while it is negative if a team allows more runs than it scores.
# Run differential can be used to predict the expected win total for a team.
# The differences in records between close games and blowout games can deviate the actual win–loss record from expected record based on run differential.

# Read games logs from the 2017 season into a dataframe
input_df = pd.read_table(fp+"GL2017.TXT", sep=",", header=None)


# warnings.filterwarnings(action="ignore")
import warnings
warnings.filterwarnings(action="ignore")

# Method to rename columns of an input dataframe (for readability)
# Input type: dataframe
# Output type: dataframe
def rename_cols(input_df):
    input_df.rename(columns = {0: 'Date', 2: 'Day', 3: 'Visiting Team', 4: 'League', 6: 'Home Team', 9: 'Runs Visitor', 10: 'Runs Home'}, inplace=True)
    return input_df

# Invoke function to rename columns
input_df = rename_cols(input_df)

# # Display
# print(input_df.head())

# Method to add new columns to indicate whether home team or visiting team won the game
# Input type: dataframe
# Output type: dataframe
def add_new_cols(input_df):
    input_df['Home Win'] = (input_df['Runs Home'] > input_df['Runs Visitor'])
    input_df['Visitor Win'] = (input_df['Runs Visitor'] > input_df['Runs Home'])
    return input_df

# Invoke method to add new columns
input_df = add_new_cols(input_df)

# Display
# print(input_df.head())

# Method to group data by home team and compute relevant statistics
# Input type: dataframe
# Output type: dataframe (with stats grouped by home team)
def proc_home_team_data(input_df):

    # Group by home team
    home_group = input_df.groupby(input_df['Home Team'])

    # Compute stats: Number of games, runs scored, runs conceded, wins, run differential
    home_df = home_group[['Runs Visitor', 'Runs Home', 'Home Win']].apply(sum)
    home_df['Home Games'] = home_group['Home Win'].count()
    home_df.rename(columns = {'Runs Visitor': 'Runs by Visitor', 'Runs Home': 'Runs at Home', 'Home Win': 'Wins at Home'}, inplace=True)
    home_df['RD at Home'] = home_df['Runs at Home'] - home_df['Runs by Visitor']
    home_df.index.rename('Team', inplace=True)
    home_df.reset_index(inplace=True)

    return home_df

# Invoke method to group data by home team and compute statistics
home_df = proc_home_team_data(input_df)

# Display
# print(home_df.head())


# Method to group data by visiting team and compute relevant statistics
# Input type: dataframe
# Output type: dataframe (with stats grouped by visiting team)
def proc_visiting_team_data(input_df):

    # Group by visiting team
    visit_group = input_df.groupby(input_df['Visiting Team'])

    # Compute stats: Number of games, runs scored, runs conceded, wins, run differential
    visit_df = visit_group[['Runs Visitor', 'Runs Home', 'Visitor Win']].apply(sum)
    visit_df['Road Games'] = visit_group['Visitor Win'].count()
    visit_df.rename(columns = {'Runs Visitor': 'Runs as Visitor', 'Runs Home': 'Runs by Home',
                               'Visitor Win': 'Wins as Visitor'}, inplace=True)
    visit_df['RD as Visitor'] = visit_df['Runs as Visitor'] - visit_df['Runs by Home']
    visit_df.index.rename('Team', inplace=True)
    visit_df.reset_index(inplace=True)

    return visit_df

# Invoke method to group data by visiting team and compute statistics
visit_df = proc_visiting_team_data(input_df)

# Display
# print(visit_df.head())

# Method to merge dataframes with statistics grouped by home and visiting teams
# and to explicitly compute explanatory and response variables
# Input type: dataframe, dataframe
# Output type: dataframe
def merge_data_frames(home_df, visit_df):
    # Compute explanatory and response variables
    overall_df = home_df.merge(visit_df, how='outer', left_on='Team', right_on='Team')
    overall_df['RD'] = overall_df['RD at Home'] + overall_df['RD as Visitor']
    overall_df['Win Pct'] = (overall_df['Wins at Home'] + overall_df['Wins as Visitor']) / (overall_df['Home Games'] + overall_df['Road Games']) * 100

    # Return dataframe with explanatory and response variables
    return overall_df

# Invoke method to merge home and visitor dataframes
overall_df = merge_data_frames(home_df, visit_df)

# # Display
# print(overall_df.head())


# Method to collate all data preprocessing steps
# Input type: dataframe
# Output type: dataframe
def extract_linear_reg_inputs(input_df):
    # Rename columns
    input_df = rename_cols(input_df)

    # Add new columns
    input_df = add_new_cols(input_df)

    # Group and process data by home team
    home_df = proc_home_team_data(input_df)

    # Group and process data by visiting team
    visit_df = proc_visiting_team_data(input_df)

    # Merge home and visitor dataframes
    overall_df = merge_data_frames(home_df, visit_df)

    return overall_df



# Get training data from 2011-2015 to train the linear regression model

# Initialize arrays to hold training data
train_run_diff = np.empty([0, 1])
train_win_pct = np.empty([0, 1])

# Loop
for year in range(2011, 2017):
    # Construct log file name
    log_file = fp+"GL" + str(year) + ".TXT"

    # Read log into a dataframe
    df = pd.read_table(log_file, sep=",", header=None)

    # Extract relevant stats into another dataframe
    df_proc = extract_linear_reg_inputs(df)

    # Add to training set
    train_run_diff = np.vstack([train_run_diff, df_proc['RD'].values.reshape([-1, 1])])
    train_win_pct = np.vstack([train_win_pct, df_proc['Win Pct'].values.reshape([-1, 1])])


lin_regr = linear_model.LinearRegression(fit_intercept=True)
lin_regr.fit(train_run_diff, train_win_pct)

# Access and display model parameters
print("Slope (a) = ", float(lin_regr.coef_), " Intercept (b) = ", float(lin_regr.intercept_))

# Get regression score (R-squared)
r_squared = lin_regr.score(train_run_diff, train_win_pct)
print("R-squared for linear fit = ", r_squared)

# Visualize
x_ax = np.array(range(int(np.min(train_run_diff)), int(np.max(train_run_diff)))).reshape(-1, 1)
y_ax = lin_regr.coef_ * x_ax + lin_regr.intercept_
# plt.ylim([0.30, 0.65])
# plt.plot([0, 0], [0.30, 0.65], "k--")
# plt.plot([-300, 300], [0.5, 0.5], "k--")
plt.plot(train_run_diff, train_win_pct, 'bo', label="training_data")
plt.plot(x_ax, y_ax, 'r', label="model_fit")
plt.xlabel("Run differential")
plt.ylabel("Win percentage")
plt.legend(loc="lower right")
# plt.show()


#Construct test dataset for 2017 season
log_file = "./gl2010_17/GL2016.TXT"
df = pd.read_table(log_file, sep=',', header=None)
df_proc = extract_linear_reg_inputs(df)
test_run_diff = df_proc['RD'].values.reshape([-1,1])
test_win_pct = df_proc['Win Pct'].values.reshape([-1,1])
predict_win_pct = lin_regr.predict(test_run_diff)

mean_abs_error_test = np.mean(np.abs(predict_win_pct - test_win_pct))
print("Percentage error on test set = ", 100. * mean_abs_error_test, "%")

#Compute percentage error for linear regression model on training set
model_fit_train = lin_regr.predict(train_run_diff)
mean_abs_error_training = np.mean(np.abs(model_fit_train - train_win_pct))
print("Percentage error on training set ", 100. * mean_abs_error_training)

plt.plot([0.35, 0.7], [0.35, 0.7], 'r')
plt.xlabel("Actual win percentage")
plt.ylabel("Predicted win percentage")
plt.title("MLB 2016 season")
# plt.show()


print(np.max(test_run_diff))
# np.argmax(
# df_proc[]



