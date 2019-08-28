#!/usr/bin/env python
# coding: utf-8

# Import Packages

# In[1]:


import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import six 

from vega import VegaLite
from enum import Enum


# Common Fields

# In[2]:


protein_profiles = "Preprocessed_Data/protein_expression.csv"

mRNA_All_minmax = "Preprocessed_Data/mRNA_All_minmax.csv"
mRNA_All_zscores = "Preprocessed_Data/mRNA_All_zscores.csv"
mRNA_All_TMM = "Preprocessed_Data/mRNA_All_TMM.csv"
mRNA_All_unnormalized = "Preprocessed_Data/mRNA_All_unnormalized.csv"

protein_interactions = "Preprocessed_Data/ProteinInteractionPartners.csv"
gene_mutations = "Preprocessed_Data/Gene_Mutations.csv"

results = "Results/All_Results.csv"
zscore_best_results = "Results/ZScore_Best_Results.json"
minmax_best_results = "Results/MinMax_Best_Results.json"
zscore_average_results = "Results/ZScore_Average_Results.json"
minmax_average_results = "Results/MinMax_Average_Results.json"

label_font = {'family': 'serif', 'color':  'black', 'size': 12}
title_font = {'family': 'serif', 'color':  'black', 'weight': 'bold', 'size': 15}


# In[3]:


Original_Proteins = Enum('Original_Proteins', 
                'RB1 CDH1 PTEN BRCA2 CDKN2A TP53')
All_Proteins = Enum('All_Proteins', 
                    'RB1 CDH1 PTEN BRCA2 CDKN2A TP53 CDH2 CDH3 CTNNB1 CCNE1 CCND1 ERBB2 ERBB3')


# In[4]:


Predictor_Type = Enum('Predictor_Type', 'Zero Mean LinearRegression ElasticNet RandomForests')


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


def tabulate_best_results(normalize_technique, dataframe): 
    linear_reg = Predictor_Type.LinearRegression.name
    selfmRNA = Scenario.selfmRNA.name
    test = "Test"
    
    col_names = ['Protein','LR selfmRNA Pearson', 'Best Pearson Score', 
                 'Best Classifier', 'Feature Set', 'Improvement', 'NRMSE']
    
    indices = [name for name in dir(All_Proteins) if not name.startswith('_')]
    resultant_df = pd.DataFrame('', index = range(39), columns = col_names)
    
    row_index = 0
    for protein in indices: 
        protein_subset = dataframe[(dataframe.Protein == protein) & 
                                   (dataframe['Train/Test'] == test) &
                                   (dataframe['Normalization'] == normalize_technique)]
        protein_subset = protein_subset.sort_values(by="Pearson", ascending=False)
        protein_subset = protein_subset.set_index(["Protein", "Regressor", "Case", 
                                                   "Normalization", "Train/Test"])
        resultant_df['Protein'][row_index] = protein
        resultant_df['LR selfmRNA Pearson'][row_index] = round(protein_subset.loc[(protein,
                                                                           linear_reg,
                                                                           selfmRNA,
                                                                           normalize_technique, 
                                                                           test), 'Pearson'], 4)
        best_values = protein_subset.head(3)
        for subset in range(3):
            protein, regressor, case, norm, test = best_values.index.tolist()[subset]
            resultant_df['Best Pearson Score'][row_index] = round(best_values.loc[(protein, 
                                                                                   regressor, 
                                                                                   case, norm, test), 
                                                                      'Pearson'], 4)
            resultant_df['Best Classifier'][row_index] = regressor
            resultant_df['Feature Set'][row_index] = case
            resultant_df['Improvement'][row_index] = round(best_values.loc[(protein, regressor,
                                                                            case, norm, test),
                                                                           'Pearson Improvement'],
                                                           4)
            resultant_df['NRMSE'][row_index] = round(best_values.loc[(protein, regressor, 
                                                                      case, norm, test),
                                                                     'NRMSE'], 4)            
        
            row_index = row_index + 1
    return resultant_df


# In[9]:


def tabulate_best_nrmse_results(normalize_technique, dataframe): 
    linear_reg = Predictor_Type.LinearRegression.name
    selfmRNA = Scenario.selfmRNA.name
    test = "Test"
    col_names = ['Protein','LR selfmRNA NRMSE', 'Best NRMSE Score', 
                 'Best Classifier', 'Feature Set', 'Improvement']
    
    indices = [name for name in dir(All_Proteins) if not name.startswith('_')]
    resultant_df = pd.DataFrame('', index = range(39), columns = col_names)
    
    row_index = 0
    for protein in indices: 
        protein_subset = dataframe[(dataframe['Protein'] == protein) & 
                                   (dataframe['Train/Test'] == test) &
                                   (dataframe['Normalization'] == normalize_technique)]
        protein_subset = protein_subset.sort_values(by="NRMSE", ascending=True)
        protein_subset = protein_subset.set_index(["Protein", "Regressor", "Case", 
                                                   "Normalization", "Train/Test"])
        resultant_df['Protein'][row_index] = protein
        resultant_df['LR selfmRNA NRMSE'][row_index] = round(protein_subset.loc[(protein,
                                                                           linear_reg,
                                                                           selfmRNA,
                                                                           normalize_technique, 
                                                                           test), 'NRMSE'], 4)
        best_values = protein_subset.head(3)
        for subset in range(3):
            protein, regressor, case, norm, test = best_values.index.tolist()[subset]
            resultant_df['Best NRMSE Score'][row_index] = round(best_values.loc[(protein, 
                                                                                   regressor, 
                                                                                   case, norm, test), 
                                                                      'NRMSE'], 4)
            resultant_df['Best Classifier'][row_index] = regressor
            resultant_df['Feature Set'][row_index] = case
            resultant_df['Improvement'][row_index] = round(best_values.loc[(protein, regressor,
                                                                            case, norm, test),
                                                                           'NRMSE Improvement'],
                                                           4)          
        
            row_index = row_index + 1
    return resultant_df


# In[10]:


def tabulate_average_results(normalize_technique, dataframe): 
    test = "Test"
    indices = [name for name in dir(Predictor_Type) if not name.startswith('_')]
    col_names = [name for name in dir(Scenario) if not name.startswith('_')]
    resultant_df = pd.DataFrame('', index=indices, columns=col_names)
    resultant_df.index.name = 'Predictor'
    for predictor in Predictor_Type:
        if(predictor == Predictor_Type.Zero or predictor == Predictor_Type.Mean):
            continue
        
        for scenario in Scenario: 
            subset = dataframe[(dataframe['Train/Test'] == test) &
                               (dataframe.Case == scenario.name) &                           
                               (dataframe.Regressor == predictor.name) & 
                               (dataframe.Normalization == normalize_technique)]
            resultant_df[scenario.name][predictor.name] = subset.Pearson.mean()
            
    return resultant_df            


# In[11]:


def tabulate_average_results_2(normalize_technique, dataframe):
    test = "Test"
    index = 0
    col_names = ['Predictor', 'Case', 'PearsonScoreImprovement']
    resultant_df = pd.DataFrame(index=range(30), columns = col_names)
    for predictor in Predictor_Type:
        if(predictor == Predictor_Type.Zero or predictor == Predictor_Type.Mean):
            continue        
        for scenario in Scenario: 
            subset = dataframe[(dataframe['Train/Test'] == test) &
                           (dataframe.Case == scenario.name) &                           
                           (dataframe.Regressor == predictor.name) & 
                           (dataframe.Normalization == normalize_technique)]
            resultant_df['Predictor'][index] = predictor.name
            resultant_df['Case'][index] = scenario.name
            resultant_df['PearsonScoreImprovement'][index] = round(subset['Pearson Improvement'].mean(), 4)
            index = index + 1 
    return resultant_df            


# In[12]:


def tabulate_normalization_improvement(dataframe):
    colnames = ['Normalization' , 'Average Improvement']
    index = 0
    resultant_df = pd.DataFrame(index = range(3), columns = colnames)
    for technique in NormalizeTechnique:
        if(technique == NormalizeTechnique.UN):
            continue;
        subset = dataframe[(dataframe['Normalization'] == technique.name) & 
                           (dataframe['Train/Test'] == 'Test')]
        resultant_df['Normalization'][index] = technique.name
        resultant_df['Average Improvement'][index] = subset['Pearson Improvement'].max()
        index = index + 1
    return resultant_df


# In[13]:


def estimate_linear_regression_line(x, y):
    n = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x - n*mean_y*mean_x)
    SS_xx = np.sum(x*x - n*mean_x*mean_x)
    
    b1 = SS_xy / SS_xx
    b0 = mean_y - b1*mean_x
    
    x_max = np.max(x)
    y_max = b0 + b1*x_max
    x_min = -0.5
    y_min = b0 + b1*x_min
    
    return x_max, y_max, x_min, y_min


# In[70]:


def visualize_table_heatmap(title, dataframe):
    dataframe = dataframe.fillna(0)
    x = dataframe[dataframe.Predictor == 'RandomForests'].sort_values(by = "PearsonScoreImprovement",
                                                                      ascending = False)
    
    indexToSort = x.Case.to_list()
    true_value="false"
    return VegaLite({
        "title": {"text": title, "anchor": "middle"},
        "encoding": {
            "y": {"field": "Case", "type": "nominal", "title": "Feature Set", 
                  "axis": {"offset": 2}, "sort": indexToSort},
            "x": {"field": "Predictor", "type": "nominal", "title": "Estimator",
                  "axis": {"labelAngle": 0, "labelPadding": 20, "offset": 2},  
                  "sort": ["LinearRegression", "ElasticNet", "RandomForests"]}},
        "layer": [
            {
                "mark": "rect",
                "encoding": {
                    "color": {
                        "field": "PearsonScoreImprovement",
                        "type": "quantitative",
                        "title": "Mean Pearson Improvement",
                        "legend": {"direction": "horizontal", "gradientLength": 125,
                                   "orient": "bottom"}
                    }}},
            {
                "mark": "text",
                "encoding": 
                {
                    "text": {"field": "PearsonScoreImprovement", "type": "quantitative"},
                    "color": {
                        "condition": {"test": "datum['PearsonScoreImprovement'] < 0.0", 
                                      "value": "black"},
                        "value": "white", 
                        "scale": {"zero": False}}                    
                }
            }],
        "config": {
            "scale": {"bandPaddingInner": 0, "bandPaddingOuter": 0},
            "text": {"baseline": "middle"}
        }}, dataframe)


# In[15]:


def visualize_mRNA_protein_corr_mean(protein, dataframe, colour):
    pearson_corr = round(dataframe['self_mRNA'].corr(dataframe['protein_expression']), 4)
    max_x = dataframe['protein_expression'].max() + 70
    min_y = dataframe['self_mRNA'].min() + 70
    text = "r = " + str(pearson_corr)
    return VegaLite({  "layer": [
        {
            "title": protein,
            "mark": {"type": "point", "color": colour},
            "encoding": {
                "x": {"field": "self_mRNA", "type": "quantitative", 
                      "title": "self mRNA expression"},
                "y": {"field": "protein_expression", "type": "quantitative"}
            }
        },
        {
            "mark": {"type" : "text", "align": "center", "baseline": "middle",
                     "text": text, "dx": max_x, "dy": min_y}
        },
        {            
            "mark": {"type": "errorband", "extent": "stdev", "opacity": 0.2, "color": colour},
            "encoding": {
                "y": {
                    "field": "protein_expression",
                    "type": "quantitative",
                    "title": "Protein Expression"
                }
            }
        },
        {
            "mark": {"type": "rule", "color": colour},
            "encoding": {
                "y": {
                    "field": "protein_expression",
                    "type": "quantitative",
                    "aggregate": "mean"
                }
            }
        }
    ]}, dataframe)


# In[16]:


def visualize_mRNA_protein_corr(protein, dataframe, colour):
    pearson_corr = round(dataframe['self_mRNA'].corr(dataframe['protein_expression']), 4)
    max_x = dataframe['protein_expression'].max() + 70
    min_y = dataframe['self_mRNA'].min() + 70
    text = "r = " + str(pearson_corr)
    x1, y1, x2, y2  = estimate_linear_regression_line(dataframe['self_mRNA'], 
                                                      dataframe['protein_expression'])
    return VegaLite({  "layer": [
        {
            "title": protein,
            "mark": {"type": "point", "color": colour},
            "encoding": {
                "x": {"field": "self_mRNA", "type": "quantitative", 
                      "title": "self mRNA expression"},
                "y": {"field": "protein_expression", "type": "quantitative",
                      "title": "Protein expression"}
            }
        },
        {
            "mark": {"type" : "text", "align": "center", "baseline": "middle",
                     "text": text, "dx": max_x, "dy": min_y}
        },
        {
            "data": {
                "values": [
                    {"x": x1, "y": y1},
                    {"x": x2, "y": y2}
                ]},
            "mark": {"type": "line", "color": colour},
            "encoding": {
                "x": {"type": "quantitative", "field": "x"},
                "y": {"type": "quantitative", "field": "y"}}}]}, dataframe)


# Loading files

# In[17]:


input_features = pd.read_csv(mRNA_All_zscores, index_col=0)
input_features.head()


# In[18]:


expected_values = pd.read_csv(protein_profiles, index_col=0)
expected_values.head()


# In[19]:


result_df = pd.read_csv(results)
result_df.head()


# In[20]:


def compute_improvements(dataframe, *metrics):
    dataframe = dataframe.set_index(["Protein", "Regressor", "Case", 
                                     "Normalization", "Train/Test"])
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
                computed_improvement =  base - value
                dataframe.loc[(protein, regressor, case, norm, train_test), 
                              improvement] = computed_improvement
    return dataframe


# In[21]:


result_df_nrmse = compute_improvements(result_df, Metric.NRMSE.name)
result_df_nrmse.head()


# In[22]:


result_df_nrmse.to_csv("All_Results_NRMSE.csv")


# Tabulate Results

# In[23]:


best_zscore = tabulate_best_results(NormalizeTechnique.ZScore.name, result_df)
best_zscore.head()


# In[24]:


best_zscore.to_csv("Best_Zscore_Top3.csv")


# In[25]:


best_minmax = tabulate_best_results(NormalizeTechnique.MinMax.name, result_df)
best_minmax.head()
best_minmax.to_csv("Best_MinMax_Top3.csv")


# In[26]:


best_zscore_nrmse = tabulate_best_nrmse_results(NormalizeTechnique.ZScore.name, 
                                                result_df_nrmse.reset_index())
best_zscore_nrmse.head()


# In[27]:


best_zscore_nrmse.to_csv("Best_NRMSE_ZScore.csv")


# In[28]:


result_df.head()


# In[29]:


average_normalization_improvement = tabulate_normalization_improvement(result_df)
average_normalization_improvement.head()


# In[30]:


VegaLite({
    "title": "Average Improvement in Pearon Correlation across different Normalization",
    "mark": "bar",
    "encoding": {
        "y": {
            "field": "Average Improvement", "type": "quantitative"
        },
        "x": {
            "field": "Normalization", "type": "nominal",
            "axis": {"title": "Normalization Technique"}
        }}}, average_normalization_improvement)


# Table Heatmap

# All Proteins

# In[31]:


average_zscore_all = tabulate_average_results_2(NormalizeTechnique.ZScore.name, result_df)
average_zscore_all.head()


# In[71]:


visualize_table_heatmap("Average Improvement in Pearson Correlation", average_zscore_all)


# In[33]:


average_minmax_all = tabulate_average_results_2(NormalizeTechnique.MinMax.name, result_df)
average_minmax_all.head()


# In[34]:


visualize_table_heatmap("Average Improvement in Pearson Correlation", average_minmax_all)


# RB1

# In[35]:


data = {'self_mRNA': input_features['RB1'], 'protein_expression': expected_values['RB1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("RB1", dataframe, "#004080")


# CDH1

# In[36]:


data = {'self_mRNA': input_features['CDH1'], 'protein_expression': expected_values['CDH1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDH1", dataframe, "#006666")


# PTEN

# In[37]:


data = {'self_mRNA': input_features['PTEN'], 'protein_expression': expected_values['PTEN']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("PTEN", dataframe, "#b35900")


# BRCA2

# In[38]:


data = {'self_mRNA': input_features['BRCA2'], 'protein_expression': expected_values['BRCA2']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("BRCA2", dataframe, "#993366")


# CDKN2A

# In[39]:


data = {'self_mRNA': input_features['CDKN2A'], 'protein_expression': expected_values['CDKN2A']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDKN2A", dataframe, "#ffcc00")


# TP53

# In[40]:


data = {'self_mRNA': input_features['TP53'], 'protein_expression': expected_values['TP53']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("TP53", dataframe, "#004d66") 


# CDH2

# In[41]:


data = {'self_mRNA': input_features['CDH2'], 'protein_expression': expected_values['CDH2']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDH2", dataframe, "#800000") 


# CHD3

# In[42]:


data = {'self_mRNA': input_features['CDH3'], 'protein_expression': expected_values['CDH3']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDH3", dataframe, "#00264d") 


# ERBB2

# In[43]:


data = {'self_mRNA': input_features['ERBB2'], 'protein_expression': expected_values['ERBB2']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("ERBB2", dataframe, "#660033") 


# ERBB3

# In[44]:


data = {'self_mRNA': input_features['ERBB3'], 'protein_expression': expected_values['ERBB3']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("ERBB3", dataframe, "#330066") 


# CTNNB1

# In[45]:


data = {'self_mRNA': input_features['CTNNB1'], 'protein_expression': expected_values['CTNNB1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CTNNB1", dataframe, "#80002a") 


# CCNE1

# In[46]:


data = {'self_mRNA': input_features['CCNE1'], 'protein_expression': expected_values['CCNE1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CCNE1", dataframe, "#1c9099") 


# CCND1

# In[47]:


data = {'self_mRNA': input_features['CCND1'], 'protein_expression': expected_values['CCND1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CCND1", dataframe, "#804000") 


# In[48]:


from platform import python_version
print(python_version())


# In[49]:


input_features = pd.read_csv(mRNA_All_minmax, index_col=0)
input_features.head()


# In[ ]:





# RB1

# In[50]:


data = {'self_mRNA': input_features['RB1'], 'protein_expression': expected_values['RB1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("RB1", dataframe, "#004080")


# CDH1

# In[51]:


data = {'self_mRNA': input_features['CDH1'], 'protein_expression': expected_values['CDH1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDH1", dataframe, "#006666")


# PTEN

# In[52]:


data = {'self_mRNA': input_features['PTEN'], 'protein_expression': expected_values['PTEN']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("PTEN", dataframe, "#b35900")


# BRCA2

# In[53]:


data = {'self_mRNA': input_features['BRCA2'], 'protein_expression': expected_values['BRCA2']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("BRCA2", dataframe, "#993366")


# CDKN2A

# In[54]:


data = {'self_mRNA': input_features['CDKN2A'], 'protein_expression': expected_values['CDKN2A']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDKN2A", dataframe, "#ffcc00")


# TP53

# In[55]:


data = {'self_mRNA': input_features['TP53'], 'protein_expression': expected_values['TP53']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("TP53", dataframe, "#004d66") 


# CDH2

# In[56]:


data = {'self_mRNA': input_features['CDH2'], 'protein_expression': expected_values['CDH2']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDH2", dataframe, "#800000") 


# CHD3

# In[57]:


data = {'self_mRNA': input_features['CDH3'], 'protein_expression': expected_values['CDH3']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CDH3", dataframe, "#00264d") 


# ERBB2

# In[58]:


data = {'self_mRNA': input_features['ERBB2'], 'protein_expression': expected_values['ERBB2']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("ERBB2", dataframe, "#660033") 


# ERBB3

# In[59]:


data = {'self_mRNA': input_features['ERBB3'], 'protein_expression': expected_values['ERBB3']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("ERBB3", dataframe, "#330066") 


# CTNNB1

# In[60]:


data = {'self_mRNA': input_features['CTNNB1'], 'protein_expression': expected_values['CTNNB1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CTNNB1", dataframe, "#80002a") 


# CCNE1

# In[61]:


data = {'self_mRNA': input_features['CCNE1'], 'protein_expression': expected_values['CCNE1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CCNE1", dataframe, "#1c9099") 


# CCND1

# In[62]:


data = {'self_mRNA': input_features['CCND1'], 'protein_expression': expected_values['CCND1']}
dataframe = pd.DataFrame(data).reset_index(drop=True)
visualize_mRNA_protein_corr("CCND1", dataframe, "#804000") 

