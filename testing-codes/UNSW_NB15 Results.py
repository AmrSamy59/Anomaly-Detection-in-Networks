# Load the training and testing datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the training and testing datasets
unsw_train = pd.read_csv("UNSW_NB15_training-set.csv")
unsw_test = pd.read_csv("UNSW_NB15_testing-set.csv")

# Combine the datasets for analysis
unsw_data = pd.concat([unsw_train, unsw_test], ignore_index=True)

# Encode categorical features
categorical_cols = ['proto', 'service', 'state']
le = LabelEncoder()
for col in categorical_cols:
    unsw_data[col] = le.fit_transform(unsw_data[col])

# Feature scaling
scaler = StandardScaler()
unsw_data_scaled = scaler.fit_transform(unsw_data.drop(['label', 'attack_cat'], axis=1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    unsw_data_scaled, unsw_data['label'], test_size=0.3, random_state=42
)


from scipy.stats import f_oneway, chi2_contingency

# ANOVA on source bytes (sbytes) across attack categories
groups = [unsw_data[unsw_data['label'] == label]['sbytes'] for label in unsw_data['label'].unique()]
anova_result = f_oneway(*groups)
statistics, p_value = anova_result
print("ANOVA Result: statistic=", statistics, "P-value=", p_value)

# Chi-Square test on protocol type (proto) and label
contingency_table = pd.crosstab(unsw_data['proto'], unsw_data['label'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Test Result:", chi2, "P-value:", p)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Random Forest - Accuracy:", accuracy)
print("Precision:", precision, "Recall:", recall, "F1 Score:", f1)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importance
importances = rf_model.feature_importances_
feature_names = unsw_data.drop(['label', 'attack_cat'], axis=1).columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance - Random Forest (UNSW-NB15)")
plt.show()

from sklearn.metrics import confusion_matrix


# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
