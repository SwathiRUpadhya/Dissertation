#!/usr/bin/env python
# coding: utf-8

# ### Predicting protein abundance using genomic and transcriptomic profiles

# ##### Name: Swathi Ramachandra Upadhya
# ##### ID: 18200264

# Import packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn import preprocessing


# Common fields 

# In[2]:


RPPA_file="brca_tcga_pub2015\data_rppa.txt"
mRNA_profile_file="brca_tcga_pub2015\data_RNA_Seq_v2_expression_median.txt"
DNA_profile_file="brca_tcga_pub2015\data_linear_CNA.txt"
DNA_mutations_file="brca_tcga_pub2015\data_mutations_extended.txt"

protein_expression="Preprocessed_Data\protein_expression.csv"
gene_mutations = "Preprocessed_Data\Gene_Mutations.csv"
mRNA_DNA_expression_minmax="Preprocessed_Data\mRNA_DNA_minmax.csv"
mRNA_DNA_expression_zscores="Preprocessed_Data\mRNA_DNA_zscores.csv"
mRNA_DNA_expression_unnormalized="Preprocessed_Data\mRNA_DNA_unnormalized.csv"

mRNA_All_minmax = "Preprocessed_Data\mRNA_All_minmax.csv"
mRNA_All_zscores = "Preprocessed_Data\mRNA_All_zscores.csv"
mRNA_All_unnormalized = "Preprocessed_Data\mRNA_All_unnormalized.csv"
mRNA_All_TMM = "Preprocessed_Data\mRNA_All_TMM.csv"

mRNA_1000_minmax = "Preprocessed_Data\mRNA_1000_minmax.csv"
mRNA_1000_zscores = "Preprocessed_Data\mRNA_1000_zscores.csv"
mRNA_1000_unnormalized = "Preprocessed_Data\mRNA_1000_unnormalized.csv"
mRNA_1000_TMM = "Preprocessed_Data\mRNA_1000_TMM.csv"


# Loading files

# In[3]:


#Loading the file containing protein profiles
protein_profile_df = pd.read_csv(RPPA_file, sep='\t', index_col=0)
protein_profile_df.head()


# In[4]:


#Loading file containing mRNA profiles
mRNA_profile_df = pd.read_csv(mRNA_profile_file, sep='\t', index_col=0)
mRNA_profile_df.head()


# In[5]:


#Loading the file containing protein profiles
DNA_profile_df = pd.read_csv(DNA_profile_file, sep='\t', index_col=0)
DNA_profile_df.head()


# Data Cleaning and Wrangling

# Working with Protein expressions 

# In[6]:


protein_profile_df.info()


# In[7]:


protein_profile_df.shape


# In[8]:


protein_profile_df.isnull().sum()


# In[9]:


protein_profile_df.dropna(inplace=True)
protein_profile_df.shape


# In[10]:


protein_profile_df = protein_profile_df.transpose()
protein_profile_df.head()


# Working with mRNA expressions

# In[11]:


mRNA_profile_df.info()


# In[12]:


mRNA_profile_df.shape


# In[13]:


mRNA_profile_df.isnull().sum()


# In[14]:


#Dropping rows which contain null values 
mRNA_profile_df.dropna(inplace=True)
#Checking the shape after dropping the rows with null values
mRNA_profile_df.shape


# In[15]:


#Transposing the data frame to get the list of patients as columns
mRNA_profile_df = mRNA_profile_df.transpose()
mRNA_profile_df.head()


# Working with DNA expressions

# In[16]:


DNA_profile_df.info()


# In[17]:


DNA_profile_df.shape


# In[18]:


DNA_profile_df.isnull().sum()


# In[19]:


DNA_profile_df.dropna(inplace=True)
DNA_profile_df.shape


# In[20]:


DNA_profile_df = DNA_profile_df.transpose()
DNA_profile_df.head()


# Finding list of common patients to be used for prediction

# In[21]:


patients = np.intersect1d(protein_profile_df.index, mRNA_profile_df.index)
common_patients = np.intersect1d(DNA_profile_df.index, patients)
print("Number of common patients: ", len(common_patients))
print (common_patients)


# In[22]:


protein_profile_subset = protein_profile_df[protein_profile_df.index.isin(common_patients)]
print("Shape:" , protein_profile_subset.shape)


# In[23]:


#Sorting the index in order to have the same order in both the dataframes
protein_profile_subset = protein_profile_subset.sort_index()
protein_profile_subset.head()


# In[24]:


#Sorting the index in order to have the same order in both the dataframes
mRNA_profile_subset = mRNA_profile_df[mRNA_profile_df.index.isin(common_patients)]
print("Shape:" , mRNA_profile_subset.shape)


# In[25]:


mRNA_profile_subset = mRNA_profile_subset.sort_index()
mRNA_profile_subset.head()


# In[26]:


#Sorting the index in order to have the same order in both the dataframes
DNA_profile_subset = DNA_profile_df[DNA_profile_df.index.isin(common_patients)]
print("Shape:" , DNA_profile_subset.shape)


# In[27]:


DNA_profile_subset = DNA_profile_subset.sort_index()
DNA_profile_subset.head()


# Selecting required protein

# In[28]:


protein_expression_df = protein_profile_subset[['RB1|Rb', 'CDH1|E-Cadherin',
                                                'PTEN|PTEN', 'BRCA2|BRCA2',
                                                'CDKN2A|P16INK4A;CDKN2A|p16_INK4a', 
                                                'TP53|p53', 'CTNNB1|beta-Catenin',
                                                'CCNE1|Cyclin_E1', 'CCND1|Cyclin_D1',
                                                'CDH2|N-Cadherin', 'CDH3|P-Cadherin',
                                                'ERBB2|HER2', 'ERBB3|HER3'
                                               ]]
protein_expression_df.columns = ['RB1', 'CDH1', 'PTEN', 'BRCA2', 'CDKN2A', 
                                 'TP53', 'CTNNB1', 'CCNE1', 'CCND1', 'CDH2',
                                 'CDH3', 'ERBB2', 'ERBB3']
protein_expression_df.head()


# In[29]:


protein_expression_df.to_csv(protein_expression)


# DNA Mutations

# In[30]:


#Loading file containing mRNA profiles
DNA_mutations_df = pd.read_csv(DNA_mutations_file, sep='\t')
DNA_mutations_df.head()


# In[31]:


proteins = protein_expression_df.columns.tolist()
DNA_mutations_subset = DNA_mutations_df[DNA_mutations_df.Variant_Classification != 'Silent']
DNA_mutations_subset = DNA_mutations_subset[['Hugo_Symbol', 'Tumor_Sample_Barcode']]
DNA_mutations_subset = DNA_mutations_subset[DNA_mutations_subset['Hugo_Symbol'].isin(proteins)]
DNA_mutations_subset = DNA_mutations_subset.reset_index(drop=True)
print('Shape:\n', DNA_mutations_subset.shape)
DNA_mutations_subset.head()


# In[32]:


#Creating an empty dataframe of the desired format
mutations = pd.DataFrame('0', mRNA_profile_subset.index, proteins)
mutations.head()


# In[33]:


#Generating mutations data
#Obtaining the column names 
columns = list(DNA_mutations_subset)
#Obtaining the list of existing common patients
tumour_samples = mRNA_profile_subset.index.tolist()
#Updating the mutations dataframe using the DNA mutations data
for index in range(DNA_mutations_subset.shape[0]):
    #Obtaining the patients which is the row in our resulting mutations dataframe
    row_index = DNA_mutations_subset[columns[1]][index]
    #Obtaining the protein which is the column index in our resulting mutations dataframe
    column_index = DNA_mutations_subset[columns[0]][index]
    if(row_index in tumour_samples):
        mutations[column_index][row_index] = 1
mutations.head()


# In[34]:


mutations.to_csv(gene_mutations)


# #### Normalization Techniques

# Z_scores data

# In[35]:


#Applying ZScore normalization on DNA data 
DNA_profile_zscore = DNA_profile_subset.apply(zscore)
DNA_profile_zscore.head()


# In[36]:


#Applying Zscore normalization on mRNA Data 
mRNA_profile_zscore = mRNA_profile_subset.apply(zscore)
mRNA_profile_zscore.head()


# In[37]:


#Since mRNA Zscore dataframe contains some null values, 
#we intend to drop the cells containing null values 
mRNA_zScore_transpose = mRNA_profile_zscore.transpose()
#Dropping rows which contain null values 
mRNA_zScore_transpose.dropna(inplace=True)
#Checking the shape after dropping the rows with null values
print("Shape: ", mRNA_zScore_transpose.shape)
mRNA_zScore_transpose.head()


# In[38]:


mRNA_zScore = mRNA_zScore_transpose.transpose()


# In[39]:


mRNA_zScore.columns


# Min-Max normalization

# In[40]:


def perform_minmax_normalization(dataframe): 
    min_max_scaler = preprocessing.MinMaxScaler()
    df_values = dataframe.values #returns a numpy array
    df_scaled = min_max_scaler.fit_transform(df_values)
    df_min_max_scaled = pd.DataFrame(df_scaled, 
                                     index = dataframe.index, 
                                     columns = dataframe.columns)
    return df_min_max_scaled


# In[41]:


mRNA_min_max_scaled = perform_minmax_normalization(mRNA_profile_subset)
mRNA_min_max_scaled.head()


# In[42]:


DNA_min_max_scaled = perform_minmax_normalization(DNA_profile_subset)
DNA_min_max_scaled.head()


# #### Store pre-processed data

# mRNA and DNA expressions combined 

# In[43]:


def select_mRNA_DNA_profiles(mRNA_profiles, DNA_profiles):
    data = {'mRNA_RB1': mRNA_profiles['RB1'], 'DNA_RB1': DNA_profiles['RB1'], 
            'mRNA_CDH1': mRNA_profiles['CDH1'], 'DNA_CDH1': DNA_profiles['CDH1'],
            'mRNA_PTEN': mRNA_profiles['PTEN'], 'DNA_PTEN': DNA_profiles['PTEN'],
            'mRNA_BRCA2': mRNA_profiles['BRCA2'], 'DNA_BRCA2': DNA_profiles['BRCA2'],
            'mRNA_CDKN2A': mRNA_profiles['CDKN2A'], 'DNA_CDKN2A': DNA_profiles['CDKN2A'],
            'mRNA_TP53': mRNA_profiles['TP53'], 'DNA_TP53': DNA_profiles['TP53'],
            'mRNA_CTNNB1': mRNA_profiles['CTNNB1'], 'DNA_CTNNB1': DNA_profiles['CTNNB1'],
            'mRNA_CCNE1': mRNA_profiles['CCNE1'], 'DNA_CCNE1': DNA_profiles['CCNE1'],
            'mRNA_CCND1': mRNA_profiles['CCND1'], 'DNA_CCND1': DNA_profiles['CCND1'],
            'mRNA_CDH2': mRNA_profiles['CDH2'], 'DNA_CDH2': DNA_profiles['CDH2'],
            'mRNA_CDH3': mRNA_profiles['CDH3'], 'DNA_CDH3': DNA_profiles['CDH3'],
            'mRNA_ERBB2': mRNA_profiles['ERBB2'], 'DNA_ERBB2': DNA_profiles['ERBB2'],
            'mRNA_ERBB3': mRNA_profiles['ERBB3'], 'DNA_ERBB3': DNA_profiles['ERBB3'],
           } 
    
    combined_dataframe = pd.DataFrame(data, index = mRNA_profiles.index)
    return combined_dataframe


# In[44]:


#Zscore
mRNA_DNA_expression_zscore = select_mRNA_DNA_profiles(mRNA_zScore, DNA_profile_zscore)
mRNA_DNA_expression_zscore.to_csv(mRNA_DNA_expression_zscores)
mRNA_DNA_expression_zscore.head()


# In[45]:


#Unnormalized
mRNA_DNA_unnormalized_data = select_mRNA_DNA_profiles(mRNA_profile_subset, DNA_profile_subset)
mRNA_DNA_unnormalized_data.to_csv(mRNA_DNA_expression_unnormalized)
mRNA_DNA_unnormalized_data.head()


# In[46]:


#Min-max normalization
mRNA_DNA_minmax = select_mRNA_DNA_profiles(mRNA_min_max_scaled, DNA_min_max_scaled)
mRNA_DNA_minmax.to_csv(mRNA_DNA_expression_minmax)
mRNA_DNA_minmax.head()


# mRNA profiles of 20,000 proteins

# In[47]:


#Zscores
mRNA_zScore.to_csv(mRNA_All_zscores)


# In[48]:


#Unnormalized 
mRNA_profile_subset.to_csv(mRNA_All_unnormalized)


# In[49]:


#Min-Max normalized
mRNA_min_max_scaled.to_csv(mRNA_All_minmax)

