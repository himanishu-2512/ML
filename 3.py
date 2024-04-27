import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Random Forest classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(x_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Calculate accuracy
score = accuracy_score(y_pred, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Get feature importances
feature_importances = pd.Series(model.feature_importances_, index=np.arange(data.shape[1])).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()

# import pandas as pd

# Get feature importances
feature_importances = pd.Series(model.feature_importances_, index=np.arange(data.shape[1])).sort_values(ascending=False)

# Create a DataFrame for feature importances
feature_importance_table = pd.DataFrame({'Feature': feature_importances.index, 'Importance': feature_importances.values})

print("Feature Importance Table:")
print(feature_importance_table)

from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test, y_pred, target_names=np.unique(labels), output_dict=True)

# Convert report to DataFrame
performance_metrics_table = pd.DataFrame(report).transpose()

# Convert confusion matrix to DataFrame
confusion_matrix_table = pd.DataFrame(cm, index=np.unique(labels), columns=np.unique(labels))

print("Confusion Matrix Table:")
print(confusion_matrix_table)


print("Performance Metrics Table:")
print(performance_metrics_table)


# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
