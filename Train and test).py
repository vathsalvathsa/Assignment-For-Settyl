import pandas as pd

# Load the CSV file
df = pd.read_csv('D:/settel.csv')

# Iterate over all columns in the DataFrame
for col in df.columns:
    # Attempt to convert each value in the column to float
    try:
        df[col] = df[col].astype(float)
    except ValueError:
        # If conversion to float fails, leave the column unchanged
        pass

# Save the updated DataFrame with float values to a new CSV file
df.to_csv('updated_csv_file.csv', index=False)

#Train and Test
from sklearn.model_selection import train_test_split
X = df.drop(columns=['externalStatus'])
y = df['externalStatus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


