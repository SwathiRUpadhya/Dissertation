#!/usr/bin/env python
# coding: utf-8

# In[10]:


import requests
import json, os
import pandas as pd


# In[11]:


string_api_url = "https://string-db.org/api"
output_format = "json"
method = "interaction_partners"
limit = 1000
protein_details = []


# In[12]:


genes = ["RB1","CDH1","PTEN","BRCA2","CDKN2A","TP53","CTNNB1",
         "CCNE1","CCND1", "CDH2", "CDH3", "ERBB2", "ERBB3"]
species = "9606"


# In[13]:


temp_json_file="ProteinInteractions.json"
protein_interaction_partners_file = "Preprocessed_Data/ProteinInteractionPartners.csv"


# In[14]:


request_url = string_api_url + "/" + output_format + "/" + method + "?"
request_url += "identifiers=%s" % "%0d".join(genes)
request_url += "&" + "species=" + species
request_url += "&" + "limit=" + str(limit)
print(request_url)
response = requests.get(request_url)
protein_details.append(response.json())


# In[15]:


with open(temp_json_file, 'w') as file:
    json.dump(protein_details, file, indent=4)


# In[16]:


#Loading the movie details on to a data frame
with open(temp_json_file, "r") as file:
    data = json.load(file)
os.remove(temp_json_file)


# In[17]:


combined_lists = data[0]


# In[18]:


protein_partners = pd.DataFrame.from_dict(combined_lists)
protein_partners.head()


# In[19]:


protein_partners.to_csv(protein_interaction_partners_file)

