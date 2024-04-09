import pandas as pd

# Load the CSV file
df = pd.read_csv('C:/Users/Sundram Vaths/Desktop/Assignment For Settyl/updated_csv_file.csv')

# Convert all columns from string to float
df = df.apply(pd.to_numeric, errors='coerce')

# Check if conversion was successful
print(df.dtypes)
# Save the converted CSV file
df.to_csv('C:/Users/Sundram Vaths/Desktop/Assignment For Settyl/updated_csv_file.csv', index=False)

