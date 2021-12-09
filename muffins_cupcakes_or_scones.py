#!/usr/bin/env python
# coding: utf-8

# **Classifying Muffins and Cupcakes with SVM**

# __Step 1:__ Import Packages

# In[1]:


# Packages for analysis
import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Pickle package
import pickle


# __Step 2:__ Import Data

# In[2]:


# Read in muffin and cupcake ingredient data
recipes = pd.read_csv('recipes_muffins_cupcakes.csv')
recipes


# __Step 3:__ Prepare the Data

# In[3]:


# Plot two ingredients
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type',
           palette='Set2', fit_reg=False, scatter_kws={"s": 70});


# In[4]:


# Specify inputs for the model
# ingredients = recipes[['Flour', 'Milk', 'Sugar', 'Butter', 'Egg', 'Baking Powder', 'Vanilla', 'Salt']].as_matrix()
ingredients = recipes[['Flour','Sugar']].values
type_label = np.where(recipes['Type']=='Muffin', 0, 1)

# Feature names
recipe_features = recipes.columns.values[1:].tolist()
recipe_features


# __Step 4:__ Fit the Model

# In[5]:


# Fit the SVM model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)


# __Step 5:__ Visualize Results

# In[6]:


# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# In[7]:


# Plot the hyperplane
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set2', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black');


# In[8]:


# Look at the margins and support vectors
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set2', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none');


# __Step 6:__ Predict New Case

# In[9]:


# Create a function to guess when a recipe is a muffin or a cupcake
def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('You\'re looking at a muffin recipe!')
    else:
        print('You\'re looking at a cupcake recipe!')


# In[10]:


# Predict if 50 parts flour and 20 parts sugar
muffin_or_cupcake(50, 20)


# In[11]:


# Plot the point to visually see where the point lies
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', palette='Set2', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50, 20, 'yo', markersize='9');


# In[12]:


# Predict if 40 parts flour and 20 parts sugar
muffin_or_cupcake(40,20)


# In[13]:


muffin_cupcake_dict = {'muffin_cupcake_model': model, 'muffin_cupcake_features': ['Flour','Sugar'], 'all_features': recipe_features}


# In[14]:


muffin_cupcake_dict


# In[15]:


# Pickle
pickle.dump(muffin_cupcake_dict, open("muffin_cupcake_dict.p", "wb"))


# In[16]:


# S = String
pickle.dumps(muffin_cupcake_dict)


# ## Classifying Muffins, Cupcakes, and Scones with SVM 

# In[17]:


#Packages have already been imported in Step 1 of "Classifying Muffins and Cupcakes with SVM"


# ### Step 1: Import Data 

# In[24]:


# Read in muffins, cupcakes, or scones ingredient data
recipes_2 = pd.read_csv('recipes_muffins_cupcakes_scones_2.csv')
recipes_2


# ### Step 2: Data Prep 

# In[25]:


# Plot two ingredients
sns.lmplot(x='Flour', y='Sugar', data=recipes_2, hue='Type',
           palette='Set2', fit_reg=False, scatter_kws={"s": 70});


# In[26]:


# Specify inputs for the model
# ingredients = recipes_2[['Flour', 'Milk', 'Sugar', 'Butter', 'Egg', 'Baking Powder', 'Vanilla', 'Salt']].value
ingredients = recipes_2[['Flour','Sugar']].values
type_label = np.where(recipes_2['Type']=='Muffin', 0, np.where(recipes_2['Type']=='Scone', 2, 1))

# Feature names
recipe_2_features = recipes_2.columns.values[1:].tolist()
recipe_2_features


# ### Step 3: Fitting the Model 

# In[27]:


# Fit the SVM model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)


# ### Step 4: Visualizing the Model 

# In[28]:


# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# In[29]:


# Plot the hyperplane
sns.lmplot(x='Flour', y='Sugar', data=recipes_2, hue='Type', palette='Set2', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black');


# In[30]:


# Look at the margins and support vectors
sns.lmplot(x='Flour', y='Sugar', data=recipes_2, hue='Type', palette='Set2', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none');


# ### Step 5: Predicting a Muffin, Cupcake, or Scone 

# In[31]:


#a function to guess when a recipe is a muffin, cupcake, or a scone
def muffin_cupcake_or_scone(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('This is a muffin recipe!')
    elif(model.predict([[flour, sugar]]))==1:
        print('This is a cupcake recipe!')
    else:
        print('This is a scone recipe!')


# In[32]:


# Predict if 60 parts flour and 30 parts sugar makes a muffin, cupcake, or scone
muffin_cupcake_or_scone(60, 30)


# In[33]:


# Predict if 40 parts flour and 15 parts sugar makes a muffin, cupcake, or scone
muffin_cupcake_or_scone(40, 15)


# In[34]:


# Predict if 20 parts flour and 35 parts sugar makes a muffin, cupcake, or scone
muffin_cupcake_or_scone(20, 35)


# In[35]:


# Plot the point to visually see where the point lies
sns.lmplot(x='Flour', y='Sugar', data=recipes_2, hue='Type', palette='Set2', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50, 20, 'yo', markersize='9');


# ## Classifying Muffins, Cupcakes, and Scones with Decision Trees 

# ### Step 1: Import Packages 

# In[37]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To create and visualize the decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


# ### Step 2: Training the Model 

# In[38]:


#fitting the model on x=ingredients and y=type_label

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(ingredients, type_label)


# ### Step 3: Visualize the Decision Tree 

# In[41]:


# Using export_graphviz to show the decision tree

export_graphviz(tree_clf,
                out_file="/Users/chrismcmanus/Downloads/muffin-cupcake-master-2/recipes_muffins_cupcakes_scones_2.csv",
                feature_names=['Flour', 'Sugar'],
                class_names=['Muffin', 'Cupcake', 'Scone'],
                rounded=True,
                filled=True
)

Source.from_file("/Users/chrismcmanus/Downloads/muffin-cupcake-master-2/recipes_muffins_cupcakes_scones_2.csv")


# In[ ]:




