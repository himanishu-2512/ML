import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import tree
import numpy as np
from scipy.stats import randint

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=3)
model.fit(x_train, y_train)



# Create a random forest classifier


# Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(model, 
#                                  param_distributions = param_dist, 
#                                  n_iter=20, 
#                                  cv=20)

# Fit the random search object to the data
# rand_search.fit(x_train, y_train)
# Create a variable for the best model
# best_rf = rand_search.best_estimator_

# Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)
# Visualize an individual decision tree from the Random Forest
for i in range(0,3):
    plt.figure(figsize=(100, 100))
    tree.plot_tree(model.estimators_[i], feature_names=[f'feature_{i}' for i in range(data.shape[1])], class_names=np.unique(labels), filled=True)
    plt.savefig('rf_individualtree'+str(i)+'.png')

# Evaluate the model on the test set
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
