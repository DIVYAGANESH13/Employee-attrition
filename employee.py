# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace 'your_dataset.csv' with your actual file)
filepath=(r"C:\Users\Jayap\OneDrive\Documents\Desktop\employee\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(filepath)
df.head()
# Assuming 'Attrition' is the target variable
target_variable = 'Attrition'

# Drop any irrelevant columns
df = df.drop(['DistanceFromHome', 'RelationshipSatisfaction','BusinessTravel'], axis=1)

# Handle categorical variables using label encoding
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Department'] = label_encoder.fit_transform(df['Department'])
df['EducationField']=label_encoder.fit_transform(df['EducationField'])
df['JobRole']=label_encoder.fit_transform(df['JobRole'])
df['MaritalStatus']=label_encoder.fit_transform(df['MaritalStatus'])
df['Over18']=label_encoder.fit_transform(df['Over18'])
df['OverTime']=label_encoder.fit_transform(df['OverTime'])
# Repeat for other categorical columns
df.head()
# Split the data into features (X) and target variable (y)
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display additional metrics
print(classification_report(y_test, y_pred))
def predict_employee_turnover(model, input_data, label_encoder):
    # Assuming 'Attrition' is the target variable
    target_variable = 'Attrition'

    # Use transform with try-except to handle new labels
    try:
        input_data['Gender'] = label_encoder.transform(input_data['Gender'])
        input_data['Department'] = label_encoder.transform(input_data['Department'])
        input_data['EducationField'] = label_encoder.transform(input_data['EducationField'])
        input_data['JobRole'] = label_encoder.transform(input_data['JobRole'])
        input_data['MaritalStatus'] = label_encoder.transform(input_data['MaritalStatus'])
        input_data['Over18'] = label_encoder.transform(input_data['Over18'])
        input_data['OverTime'] = label_encoder.transform(input_data['OverTime'])
    except ValueError as e:
        # Handle new labels as needed
        print(f"Warning: {str(e)}")
        # You might want to add logic to handle new labels here

    # Make prediction
    prediction = model.predict(input_data)

    return prediction


# Example usage:
new_employee_data = pd.DataFrame({
    'Age': [30],
    'DailyRate': [800],
    # Add other features according to your dataset
    'Gender': ['Male'],
    'Department': ['Sales'],
    'EducationField': ['Life Sciences'],
    'JobRole': ['Sales Executive'],
    'MaritalStatus': ['Single'],
    'Over18': ['Y'],
    'OverTime': ['Yes']
})

predicted_attrition = predict_employee_turnover(model, new_employee_data,label_encoder)
print(f'Predicted Attrition: {predicted_attrition[0]}')
print("Columns in training data:", X_train.columns)
print("Columns in new_employee_data:", new_employee_data.columns)
