#!/usr/bin/env python
# coding: utf-8

# Import packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt
from enum import Enum
from sklearn import ensemble
from sklearn import linear_model
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Common Fields

# In[2]:


protein_profiles = "Preprocessed_Data/protein_expression.csv"

mRNA_DNA_minmax = "Preprocessed_Data/mRNA_DNA_minmax.csv"
mRNA_DNA_TMM = "Preprocessed_Data/mRNA_DNA_TMM.csv"
mRNA_DNA_zscores = "Preprocessed_Data/mRNA_DNA_zscores.csv"
mRNA_DNA_unnormalized = "Preprocessed_Data/mRNA_DNA_unnormalized.csv"

mRNA_All_minmax = "Preprocessed_Data/mRNA_All_minmax.csv"
mRNA_All_zscores = "Preprocessed_Data/mRNA_All_zscores.csv"
mRNA_All_TMM = "Preprocessed_Data/mRNA_All_TMM.csv"
mRNA_All_unnormalized = "Preprocessed_Data/mRNA_All_unnormalized.csv"

protein_interactions = "Preprocessed_Data/ProteinInteractionPartners.csv"
gene_mutations = "Preprocessed_Data/Gene_Mutations.csv"
results = "Results/All_Results.csv"

predictors = {}
cumulative_scores = {}
random_genes = {}


# Common Enums:

# In[3]:


Predictor_Type = Enum('Predictor_Type', 'Zero Mean LinearRegression ElasticNet RandomForests')


# In[4]:


Proteins = Enum('Proteins', 
                'RB1 CDH1 CDH2 CDH3 PTEN BRCA2 CDKN2A TP53 CTNNB1 CCNE1 CCND1 ERBB2 ERBB3')


# In[5]:


NormalizeTechnique = Enum('NormalizeTechnique', 'ZScore MinMax TMM UN')


# In[6]:


Metric = Enum('Metric', 'Pearson Spearman R2 RMSE NRMSE MAPE')


# In[7]:


class Scenario(Enum):
    selfmRNA = 1
    AllmRNA_Mutations = 2
    selfmRNA_Mutations = 3
    selfmRNA_Polynomial = 4
    selfmRNA_CNA_Mutations = 5
    NRandomGenes_Mutations = 6    
    NMostVariableGenes_Mutations = 7
    AllInteractionPartners_Mutations = 8
    Top100InteractionPartners_Mutations = 9
    ThousandMostVariableGenes_Mutations = 10


# Common Methods

# In[8]:


#Computing Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[9]:


#Computing Root Mean Squared Percentage Error 
def root_mean_squared_percentage_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true))))


# In[10]:


def update_scores(annotation, scorekeeper, pearson, spearman, 
                  coeff_determination, RMSE, NRMSE, MAPE):
    protein_predictType_Pearson = ":".join([annotation, Metric.Pearson.name])
    protein_predictType_Spearman = ":".join([annotation, Metric.Spearman.name])
    protein_predictType_r2 = ":".join([annotation, Metric.R2.name])
    protein_predictType_rmse = ":".join([annotation, Metric.RMSE.name])
    protein_predictType_nrmse = ":".join([annotation, Metric.NRMSE.name])
    protein_predictType_mape = ":".join([annotation, Metric.MAPE.name])
    
    if(protein_predictType_Pearson in scorekeeper):
        pear_corr = scorekeeper[protein_predictType_Pearson] + pearson 
        scorekeeper[protein_predictType_Pearson] = pear_corr
    else:
        scorekeeper[protein_predictType_Pearson] = pearson
        
    if(protein_predictType_Spearman in scorekeeper):
        spear_corr = scorekeeper[protein_predictType_Spearman] + spearman
        scorekeeper[protein_predictType_Spearman] = spear_corr
    else:
        scorekeeper[protein_predictType_Spearman] = spearman
    
    if(protein_predictType_r2 in scorekeeper):
        cDet = scorekeeper[protein_predictType_r2] + coeff_determination
        scorekeeper[protein_predictType_r2] = cDet
    else:
        scorekeeper[protein_predictType_r2] = coeff_determination
    
    if(protein_predictType_rmse in scorekeeper):
        rmse = scorekeeper[protein_predictType_rmse] + RMSE
        scorekeeper[protein_predictType_rmse] = rmse
    else:
        scorekeeper[protein_predictType_rmse] = RMSE
        
    if(protein_predictType_nrmse in scorekeeper):
        nrmse = scorekeeper[protein_predictType_nrmse] + NRMSE
        scorekeeper[protein_predictType_nrmse] = nrmse
    else:
        scorekeeper[protein_predictType_nrmse] = NRMSE
        
    if(protein_predictType_mape in scorekeeper):
        mape = scorekeeper[protein_predictType_mape] + MAPE
        scorekeeper[protein_predictType_mape] = mape
    else:
        scorekeeper[protein_predictType_mape] = MAPE


# In[11]:


#Evaluating the predictions using different metrics
def evaluate_predictions(y_true,y_predictions):
    pearson = np.nan
    rho = np.nan
    
    if not (y_predictions == y_predictions[0]).all():
        #Compute Pearson Correlation 
        pearson, p_value = pearsonr(y_true, y_predictions)        
        #Compute Spearman Correlation 
        rho, p_value = spearmanr(y_true, y_predictions)
        
    print("\t\tPearson Correlation: ", pearson)
    print("\t\tSpearman Correlation: ", rho)
    #Compute the coefficient of determination 
    coeff_determination = r2_score(y_true, y_predictions)
    print("\t\tCoefficient of Determination: ", coeff_determination)
    #Compute Root Mean Squared Error (RMSE)
    rmse = sqrt(mean_squared_error(y_true, y_predictions))
    print("\t\tRMSE: ", rmse)
    #Compute Normalized RMSE
    nrmse = rmse/(np.max(y_true) - np.min(y_true))
    print("\t\tNRMSE: ", nrmse)
    #Compute Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true, y_predictions)
    print("\t\tMAPE: ", mape)
    
    return pearson, rho, coeff_determination, rmse, nrmse, mape


# In[12]:


def predict_and_evaluate(estimator, x, y, scorekeeper, annotation):
    predictions = estimator.predict(x)
    pcor, scor, cdet, rmse, nrmse, mape = evaluate_predictions(np.ravel(y.values), 
                                                                      predictions)
    update_scores(annotation, scorekeeper, pcor, scor, cdet, rmse, nrmse, mape)


# In[13]:


#Get or create the predictor based on the type specified 
def get_create_predictor(predictor_type): 
    if(predictor_type in predictors):
        predictor = predictors[predictor_type]
    else:
        if(predictor_type == Predictor_Type.Zero):
            predictor = DummyRegressor(strategy='constant', constant=0)
        elif(predictor_type == Predictor_Type.Mean):
            predictor = DummyRegressor(strategy='mean')
        elif(predictor_type == Predictor_Type.LinearRegression):
            predictor = linear_model.LinearRegression()
        elif(predictor_type == Predictor_Type.ElasticNet):
            predictor = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.1, random_state=0)
        else:
            predictor = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        predictors[predictor_type] = predictor
        
    return predictor


# In[14]:


def get_protein_partners(protein, dataframe=None, n_highest=None):
    protein_partners = pd.read_csv(protein_interactions, index_col=0)
    protein_partners = protein_partners.loc[protein_partners['preferredName_A'] == protein]
    if(n_highest is not None):
        protein_partners = protein_partners.sort_values("score", ascending=False).head(n_highest)
    interaction_partners = protein_partners['preferredName_B'].tolist()
    if(dataframe is None):
        return interaction_partners
    else:
        print("Protein Interaction Partners Count: ", len(interaction_partners))
        protein_columns = [col for col in dataframe.columns if col in interaction_partners]
        return protein_columns


# In[15]:


def get_most_variable_mRNAs(dataframe, n=1000):
    standard_dev = dataframe.std()
    standard_dev = standard_dev.sort_values(ascending=False)
    higheststd = standard_dev.nlargest(n, keep='all')
    return higheststd.index.values.tolist()


# In[16]:


def get_polynomial_features(X_train, X_test, degree=2):
    x_train = X_train
    x_test = X_test
    for index in range(2, degree+1):
        power = str(index)
        root = "1/" + power
        if(x_train is not None):
            x_train[power] = pd.Series()
            x_train[root] = pd.Series()
        if(x_test is not None):
            x_test[power] = pd.Series()
            x_test[root] = pd.Series()
        
        x_train[power] = X_train.abs()**(index) * np.sign(X_train)
        x_test[power] = X_test.abs()**(index) * np.sign(X_test)
        x_train[power] = X_train.abs()**(index) * np.sign(X_train)
        x_test[power] = X_test.abs()**(index) * np.sign(X_test)
        
        x_train[root] = X_train.abs()**(1/(index)) * np.sign(X_train)
        x_test[root] = X_test.abs()**(1/(index)) * np.sign(X_test)
    
    x_train.dropna(how='any', axis=0, inplace=True)
    x_test.dropna(how='any', axis=0, inplace=True)        
        
    return x_train, x_test


# In[17]:


def get_random_genes(protein, dataframe, n = 1000):
    if(protein in random_genes):
        return random_genes[protein]
    else: 
        col_names = dataframe.sample(n, axis = 1).columns.tolist()
        random_genes[protein] = col_names
        return col_names


# In[18]:


def get_sorted_protein_data(Y_train, Y_test, x_train, x_test):
    y_train = Y_train[Y_train.index.isin(x_train.index)]
    y_test = Y_test[Y_test.index.isin(x_test.index)]
    
    x_train.sort_index(inplace=True)
    x_test.sort_index(inplace=True)
    y_train.sort_index(inplace=True)
    y_test.sort_index(inplace=True)
    
    return y_train, y_test


# In[19]:


def get_train_test_data(protein, scenario, X_train, X_test, Y_train, Y_test, Z_train, Z_test):
    protein_columns = None
    if(scenario == Scenario.ThousandMostVariableGenes_Mutations):
        combined_data = [X_train, X_test]
        protein_columns = get_most_variable_mRNAs(pd.concat(combined_data))
    elif(scenario == Scenario.AllInteractionPartners_Mutations): 
        protein_columns = get_protein_partners(protein, X_train)
    elif(scenario == Scenario.Top100InteractionPartners_Mutations ):
        protein_columns = get_protein_partners(protein, X_train, n_highest=100)    
    elif(scenario == Scenario.NRandomGenes_Mutations ):
        columns_count = len(get_protein_partners(protein))
        protein_columns = get_random_genes(protein, X_test, columns_count)
    elif(scenario == Scenario.NMostVariableGenes_Mutations ):
        columns_count = len(get_protein_partners(protein))
        combined_data = [X_train, X_test]
        protein_columns = get_most_variable_mRNAs(pd.concat(combined_data), columns_count)
    elif(scenario == Scenario.AllmRNA_Mutations ):
        x_train = X_train.copy()
        x_test = X_test.copy()
    elif(scenario == Scenario.selfmRNA_Polynomial):
        x_train = X_train[[protein]].copy()
        x_test = X_test[[protein]].copy()
        x_train, x_test = get_polynomial_features(x_train, x_test, degree=3)
    elif(scenario == Scenario.selfmRNA_CNA_Mutations):
        protein_columns = [col for col in X_train.columns if protein in col]    
    else:
        x_train = X_train[[protein]].copy()
        x_test = X_test[[protein]].copy()
        
    if(protein_columns is not None):
        x_train = X_train[protein_columns].copy()
        x_test = X_test[protein_columns].copy()

    if(scenario != Scenario.selfmRNA and scenario != Scenario.selfmRNA_Polynomial):
        if(scenario != Scenario.selfmRNA_Mutations and scenario != Scenario.selfmRNA_CNA_Mutations):
            x_train[protein] = X_train[protein]
            x_test[protein] = X_test[protein]
        x_train["GeneMutation"] = Z_train[protein]
        x_test["GeneMutation"] = Z_test[protein]   
    
    print("x_train_column_count: ", len(x_train.columns))
    if(scenario != Scenario.AllmRNA_Mutations):
        print("x_train columns: ", x_train.columns.tolist())
    
    y_train, y_test = get_sorted_protein_data(Y_train[protein], Y_test[protein], x_train, x_test)
    
    return x_train, x_test, y_train, y_test


# In[20]:


def get_annotation(protein, predictor, scenario, normalize_technique, train_test):
    return ":".join([protein.name, predictor.name, scenario.name, 
                     normalize_technique.name, train_test])


# In[21]:


def get_data(normalize_technique):
    if(normalize_technique == NormalizeTechnique.ZScore):
        input_features = pd.read_csv(mRNA_All_zscores, index_col=0)
    elif(normalize_technique == NormalizeTechnique.MinMax):
        input_features = pd.read_csv(mRNA_All_minmax, index_col=0)
    elif(normalize_technique == NormalizeTechnique.TMM):
        input_features = pd.read_csv(mRNA_All_TMM, index_col=0)
    else:
        input_features = pd.read_csv(mRNA_All_unnormalized, index_col=0)
    
    #Loading the file containing protein profiles
    expected_values = pd.read_csv(protein_profiles, index_col=0)
    #Loading the file containing mutations data
    mutations_data = pd.read_csv(gene_mutations, index_col=0)
    #Returning the input features, expected values and the mutation data
    return input_features, expected_values, mutations_data 


# In[22]:


def get_mRNA_DNA_data(normalize_technique):
    if(normalize_technique == NormalizeTechnique.ZScore):
        input_features = pd.read_csv(mRNA_DNA_zscores, index_col=0)
    elif(normalize_technique == NormalizeTechnique.MinMax):
        input_features = pd.read_csv(mRNA_DNA_minmax, index_col=0)
    elif(normalize_technique == NormalizeTechnique.TMM):
        input_features = pd.read_csv(mRNA_DNA_TMM, index_col=0)
    else:
        input_features = pd.read_csv(mRNA_DNA_unnormalized, index_col=0)
    
    #Loading the file containing protein profiles
    expected_values = pd.read_csv(protein_profiles, index_col=0)
    #Loading the file containing mutations data
    mutations_data = pd.read_csv(gene_mutations, index_col=0)
    #Returning the input features, expected values and the mutation data
    return input_features, expected_values, mutations_data   


# In[23]:


def perform_kFoldCV(k, scenario, normalize_technique, X, Y, Z = None):
    Z_train = None
    Z_test = None
    kFold = KFold(n_splits=k)
    fold = 1
    for train_index, test_index in kFold.split(X, Y):
        print("\n\nFold: ", fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        Z_train, Z_test = Z.iloc[train_index], Z.iloc[test_index] 
        for protein in Proteins:
            print("\nProtein: ", protein.name)
            x_train, x_test, y_train, y_test = get_train_test_data(protein.name, scenario,
                                                                   X_train, X_test, 
                                                                   Y_train, Y_test, 
                                                                   Z_train, Z_test)
            for predictor_type in Predictor_Type:
                print("\n\tPredictor: ", predictor_type.name)
                predictor = get_create_predictor(predictor_type)      
                predictor.fit(x_train, np.ravel(y_train.values))
                predict_and_evaluate(predictor, x_train, y_train, cumulative_scores, 
                                     get_annotation(protein, predictor_type, scenario, 
                                                    normalize_technique, "Train"))
                predict_and_evaluate(predictor, x_test, y_test, cumulative_scores, 
                                     get_annotation(protein, predictor_type, scenario, 
                                                    normalize_technique, "Test"))
        fold = fold + 1


# In[24]:


def tabulate_scores(scorekeeper, num_folds):
    #printing the mean values    
    proteins = [name for name in dir(Proteins) if not name.startswith('_')]
    predictorTypes = [name for name in dir(Predictor_Type) if not name.startswith('_')]
    scenarios = [name for name in dir(Scenario) if not name.startswith('_')]
    normalizations = [name for name in dir(NormalizeTechnique) if not name.startswith('_')]
    train_test = ['Train', 'Test']
    indices = pd.MultiIndex.from_product([proteins, predictorTypes, 
                                          scenarios, normalizations, train_test],
                                         names=['Protein', 'Regressor', 'Case', 
                                                'Normalization', 'Train/Test'])
    column_names = [name for name in dir(Metric) if not name.startswith('_')]

    dataFrame = pd.DataFrame('', indices, column_names)
    
    for key, value in scorekeeper.items():
        value = value/num_folds
        parts  = key.split(":")
        dataFrame.loc[(parts[0], parts[1], parts[2], parts[3], parts[4]), parts[5]] = value
    
    return dataFrame


# In[25]:


def compute_improvements(dataframe, *metrics):
    indices = dataframe.index.tolist()
    selfmRNA = Scenario.selfmRNA.name
    linearRegression = Predictor_Type.LinearRegression.name
    for metric in metrics:
        improvement = metric + " Improvement"
        dataframe[improvement] = pd.Series()
        
        for index in indices: 
            protein, regressor, case, norm, train_test = index
            base = dataframe.loc[(protein, linearRegression , selfmRNA, norm, train_test), metric]
            value = dataframe.loc[(protein, regressor, case, norm, train_test), metric]

            if(base and value):
                computed_improvement =  value - base
                dataframe.loc[(protein, regressor, case, norm, train_test), 
                              improvement] = computed_improvement
    return dataframe


# Machine Learning Models with k-fold cross validation

# In[26]:


num_folds = 5


# self-mRNA

# ZScore

# In[27]:


scenario = Scenario.selfmRNA
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[28]:


X.head()


# In[29]:


Y.head()


# In[30]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[31]:


scenario = Scenario.selfmRNA
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[32]:


X.head()


# In[33]:


Y.head()


# In[34]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[35]:


scenario = Scenario.selfmRNA
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[36]:


X.head()


# In[37]:


Y.head()


# In[38]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[39]:


scenario = Scenario.selfmRNA
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[40]:


X.head()


# In[41]:


Y.head()


# In[42]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# self-mRNA with Mutations

# ZScore

# In[43]:


scenario = Scenario.selfmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[44]:


X.head()


# In[45]:


Y.head()


# In[46]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[47]:


scenario = Scenario.selfmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[48]:


X.head()


# In[49]:


Y.head()


# In[50]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[51]:


scenario = Scenario.selfmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[52]:


X.head()


# In[53]:


Y.head()


# In[54]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[55]:


scenario = Scenario.selfmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[56]:


X.head()


# In[57]:


Y.head()


# In[58]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Polynomial Features

# ZScore

# In[59]:


scenario = Scenario.selfmRNA_Polynomial
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[60]:


X.head()


# In[61]:


Y.head()


# In[62]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[63]:


scenario = Scenario.selfmRNA_Polynomial
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[64]:


X.head()


# In[65]:


Y.head()


# In[66]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[67]:


scenario = Scenario.selfmRNA_Polynomial
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[68]:


X.head()


# In[69]:


Y.head()


# In[70]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[71]:


scenario = Scenario.selfmRNA_Polynomial
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[72]:


X.head()


# In[73]:


Y.head()


# In[74]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# self-mRNA, CNA and Mutations

# ZScore

# In[75]:


scenario = Scenario.selfmRNA_CNA_Mutations
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_mRNA_DNA_data(normalizeTechnique)


# In[76]:


X.head()


# In[77]:


Y.head()


# In[78]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[79]:


scenario = Scenario.selfmRNA_CNA_Mutations
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_mRNA_DNA_data(normalizeTechnique)


# In[80]:


X.head()


# In[81]:


Y.head()


# In[82]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[83]:


scenario = Scenario.selfmRNA_CNA_Mutations
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_mRNA_DNA_data(normalizeTechnique)


# In[84]:


X.head()


# In[85]:


Y.head()


# In[86]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[87]:


scenario = Scenario.selfmRNA_CNA_Mutations
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_mRNA_DNA_data(normalizeTechnique)


# In[88]:


X.head()


# In[89]:


Y.head()


# In[90]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# All Interaction Partners

# ZScore

# In[91]:


scenario = Scenario.AllInteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[92]:


X.head()


# In[93]:


Y.head()


# In[94]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[95]:


scenario = Scenario.AllInteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[96]:


X.head()


# In[97]:


Y.head()


# In[98]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[99]:


scenario = Scenario.AllInteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[100]:


X.head()


# In[101]:


Y.head()


# In[102]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[103]:


scenario = Scenario.AllInteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[104]:


X.head()


# In[105]:


Y.head()


# In[106]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Top 100 Interaction Partners

# ZScore

# In[107]:


scenario = Scenario.Top100InteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[108]:


X.head()


# In[109]:


Y.head()


# In[110]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[111]:


scenario = Scenario.Top100InteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[112]:


X.head()


# In[113]:


Y.head()


# In[114]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[115]:


scenario = Scenario.Top100InteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[116]:


X.head()


# In[117]:


Y.head()


# In[118]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[119]:


scenario = Scenario.Top100InteractionPartners_Mutations 
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[120]:


X.head()


# In[121]:


Y.head()


# In[122]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# N Random Genes

# ZScore

# In[123]:


random_genes.clear()
scenario = Scenario.NRandomGenes_Mutations 
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[124]:


X.head()


# In[125]:


Y.head()


# In[126]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[127]:


random_genes.clear()
scenario = Scenario.NRandomGenes_Mutations 
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[128]:


X.head()


# In[129]:


Y.head()


# In[130]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[131]:


random_genes.clear()
scenario = Scenario.NRandomGenes_Mutations 
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[132]:


X.head()


# In[133]:


Y.head()


# In[134]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[135]:


random_genes.clear()
scenario = Scenario.NRandomGenes_Mutations 
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[136]:


X.head()


# In[137]:


Y.head()


# In[138]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# N Most Variable Genes

# ZScore

# In[139]:


scenario = Scenario.NMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[140]:


X.head()


# In[141]:


Y.head()


# In[142]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[143]:


scenario = Scenario.NMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[144]:


X.head()


# In[145]:


Y.head()


# In[146]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[147]:


scenario = Scenario.NMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[148]:


X.head()


# In[149]:


Y.head()


# In[150]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[151]:


scenario = Scenario.NMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[152]:


X.head()


# In[153]:


Y.head()


# In[154]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# 1000 Most Variable mRNAs

# ZScore

# In[155]:


scenario = Scenario.ThousandMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[156]:


X.head()


# In[157]:


Y.head()


# In[158]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[159]:


scenario = Scenario.ThousandMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[160]:


X.head()


# In[161]:


Y.head()


# In[162]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[163]:


scenario = Scenario.ThousandMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[164]:


X.head()


# In[165]:


Y.head()


# In[166]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[167]:


scenario = Scenario.ThousandMostVariableGenes_Mutations 
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[168]:


X.head()


# In[169]:


Y.head()


# In[170]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# All-mRNA

# ZScore

# In[171]:


scenario = Scenario.AllmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.ZScore
X, Y, Z = get_data(normalizeTechnique)


# In[172]:


X.head()


# In[173]:


Y.head()


# In[174]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Min-Max

# In[175]:


scenario = Scenario.AllmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.MinMax
X, Y, Z = get_data(normalizeTechnique)


# In[176]:


X.head()


# In[177]:


Y.head()


# In[178]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# TMM

# In[179]:


scenario = Scenario.AllmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.TMM
X, Y, Z = get_data(normalizeTechnique)


# In[180]:


X.head()


# In[181]:


Y.head()


# In[182]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Unnormalized

# In[183]:


scenario = Scenario.AllmRNA_Mutations 
normalizeTechnique = NormalizeTechnique.UN
X, Y, Z = get_data(normalizeTechnique)


# In[184]:


X.head()


# In[185]:


Y.head()


# In[186]:


perform_kFoldCV(num_folds, scenario, normalizeTechnique, X, Y, Z)


# Tabulate the scores

# In[187]:


dataFrame = tabulate_scores(cumulative_scores, num_folds)


# In[188]:


dataFrame = compute_improvements(dataFrame, Metric.Pearson.name, Metric.Spearman.name)


# In[189]:


dataFrame.to_csv(results)


# In[191]:


dataFrame


# In[ ]:




