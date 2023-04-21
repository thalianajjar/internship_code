import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statistics import mean, stdev

# Function to load all three predictor datasets
def load_datasets(participant_number, quest_version):
    df = pd.read_csv('data/preprocessed/preprocessed_data_' + str(participant_number) + '_v' + str(quest_version) + '.csv')
    df = df.drop(labels=['actual_day', 'actual_day.1', 'bed_time'], axis=1)

    # Oura dataset
    df_oura = pd.read_csv('data/preprocessed/preprocessed_sleep_' + str(participant_number) + '_v' + str(quest_version) + '.csv')
    df_oura = df_oura.drop(labels=['actual_day', 'bed_time'], axis=1)

    # Questionnaire dataset
    df_quest = pd.read_csv('data/preprocessed/preprocessed_questionnaires_' + str(participant_number) + '_v' + str(quest_version) + '.csv')
    df_quest = df_quest.drop(labels=['actual_day'], axis=1)

    return df, df_oura, df_quest

# Define the Lasso model used in this project
def lasso_model(X, y, random_state):
    # Standardizing the input variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Splitting train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = random_state)

    # Define model
    alphas = np.arange(0.01, 5, 0.01)
    model = LassoCV(alphas=alphas, cv=cv, max_iter=10000)

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test 

# Define the Elastic Net model used in this project
def elastic_net_model(X, y, random_state):
    # Standardizing the input variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Splitting train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = random_state)

    # Define model
    ratios = np.arange(0, 1, 0.01)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
    
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# Print the resulting model's characteristics
def print_model(model):
    # Print search results
    print('Model results\nBest alpha value: %s\n' % model.alpha_)

    # Print training results with regards to the coefficients
    print("Number of coefficients", len(model.coef_))
    print("Non-zero coeffcients", np.count_nonzero(model.coef_))
    print("Coefficients", model.coef_)

# Compute R squared and mean square error of a given model, for the training and testing sets 
def compute_model(model, X_train, y_train, X_test, y_test):
    # Training data
    r2 = model.score(X_train, y_train)
    print('R squared training set', round(r2, 3))
    adj_r2 = 1 - (1 - r2) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)
    print('Adjusted R squared training set', round(adj_r2, 3))
    pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, pred_train)
    print('MSE training set', round(mse_train, 3))

    # Test data
    r2 = model.score(X_test, y_test)
    print('\nR squared test set', round(r2, 3))
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    print('Adjusted R squared test set', round(adj_r2, 3))
    pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, pred_test)
    print('MSE test set', round(mse_test, 3))

    return pred_train, pred_test

# Return the absolute value of a value
def absvalue(value):
    return abs(value)
    
# Get all features that have a non-zero coefficient, along with their respective coefficient
def get_selected_features(X, reg):
    selected_features = []
    features_coef = []
    
    for i in range(len(reg.coef_)):
        # Only consider the variables that have not been eliminated in the L1 regularization phase
        if reg.coef_[i] != 0:
            selected_features.append(X.columns[i])
            features_coef.append(reg.coef_[i])

    return selected_features, features_coef

# Sort the selected features by order of importance
def sort_features(X, reg):
    selected_features, features_coef = get_selected_features(X, reg)
    print('Number of selected features', len(selected_features))
    print(selected_features)

    # Sort the coefficients in increasing order
    features_coef = sorted(features_coef, reverse=True, key = absvalue)
    original_coef = features_coef
    
    # Get the order of the features according to the importance of their coefficients
    indices = []
    for i in range(len(original_coef)):
        coeff = original_coef[i]
        index = features_coef.index(coeff)
        indices.append(index)

    # If two values are equal, still assign them a different index
    last_index = -1
    for i in range(len(indices)):
        if (indices[i] == last_index):
            indices[i] = indices[i-1] + 1
        else: 
            last_index = indices[i]

    # Order the names of the features accordingly
    sorted_features = []
    for i in range(len(indices)):
        index = indices.index(i)
        value = selected_features[index]
        sorted_features.append(value)
        
    return sorted_features, features_coef

# Print the equation of how to predict the variable according to the model
def print_equation(sorted_features, features_coef):
    counter = 0
    equation = ''
    for i in range(len(sorted_features)):
        if counter != 0:
            equation += ' + '
        else:
            counter += 1
        equation += '(' + str(round(features_coef[i], 5)) + ' * ' + sorted_features[i] + ')'
    print(equation)

# Runs and computes the average MSE, R squared and number of selected features of n lasso models 
def multiple_models_average(X, y, n, model_function):
    mses, r2s, adj_r2s, features_counts = [], [], [], []
    
    for i in range(n):
        model, X_train, X_test, y_train, y_test = model_function(X, y, random_state=np.random.randint(50000))
        
        # Get R-squared of model
        r2 = model.score(X_test, y_test)
        r2s.append(round(r2, 3))

        # Get adjusted R-squared of model
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
        adj_r2s.append(round(adj_r2, 3))

        # Get MSE of model
        pred_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, pred_test)
        mses.append(round(mse_test, 3))

        # Get number of selected features
        selected_features, features_coef = get_selected_features(X, model)
        features_counts.append(len(selected_features))

    print('Average R2 out of ' + str(n) + ' runs is: ' + str(round(mean(r2s), 4)) + ' ± ' + str(round(stdev(r2s), 3)))
    print('Average adjusted R2 out of ' + str(n) + ' runs is: ' + str(round(mean(adj_r2s), 4)) + ' ± ' + str(round(stdev(adj_r2s), 3)))
    print('Average MSE out of ' + str(n) + ' runs is: ' + str(round(mean(mses), 4)) + ' ± ' + str(round(stdev(mses), 3)))
    print('Average number of selected features out of ' + str(n) + ' runs is: ' + str(round(mean(features_counts), 4)) + ' ± ' + str(round(stdev(features_counts), 3)))