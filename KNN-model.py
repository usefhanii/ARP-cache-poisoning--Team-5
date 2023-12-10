import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:/Users/user/Desktop/CyberSecurity/project/hypothetical_arp_dataset2.csv' 
data = pd.read_csv(file_path)

# Separate features and labels
X = data[['Feature_1', 'Feature_2']]
y = data['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Example of real-time classification (pseudo code)
# while True:
#     new_data_point = get_new_network_data()  # Function to get new data
#     new_data_point = preprocess(new_data_point)  # Preprocess if necessary
#     prediction = knn.predict(new_data_point.reshape(1, -1))
#     print("Network state: ", "Under Attack" if prediction[0] == 1 else "Normal")

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train['Feature_1'], y=X_train['Feature_2'], hue=y_train, palette='viridis', alpha=0.6)
sns.scatterplot(x=X_test['Feature_1'], y=X_test['Feature_2'], hue=y_test, palette='coolwarm', alpha=1, edgecolor='k')

plt.title('KNN Classification - Train vs Test')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Class', loc='upper right')
plt.show()

