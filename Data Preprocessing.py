import numpy as np 
import pandas as pd

dataset = pd.read_csv('D:/settel.csv')

# Remove Duplicate 
dataset = dataset.drop_duplicates()

#Save the clean data 
dataset.to_csv('cleaned_data.csv',index = False)