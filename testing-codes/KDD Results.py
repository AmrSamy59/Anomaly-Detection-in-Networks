import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load datasets
kdd_data = pd.read_csv("kddcup.data_10_percent_corrected")

kdd_data.columns =  ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
           'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 
           'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
           'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
           'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
           'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

# Preprocessing: Label encoding for categorical features
le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    kdd_data[col] = le.fit_transform(kdd_data[col])

# Feature scaling
scaler = StandardScaler()
kdd_data_scaled = scaler.fit_transform(kdd_data.drop('label', axis=1))


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(kdd_data_scaled, kdd_data['label'], test_size=0.3, random_state=42)

from scipy.stats import f_oneway

# ANOVA on source bytes for different attack categories
groups = [kdd_data[kdd_data['label'] == label]['src_bytes'] for label in kdd_data['label'].unique()]
anova_result = f_oneway(*groups)
statistics, p_value = anova_result
print("ANOVA Result: statistic=", statistics, "P-value=", p_value)

from scipy.stats import chi2_contingency

# Chi-Square test on protocol type and label
contingency_table = pd.crosstab(kdd_data['protocol_type'], kdd_data['label'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Test Result:", chi2, "P-value:", p)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

importances = rf_model.feature_importances_
feature_names = kdd_data.drop('label', axis=1).columns

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance - Random Forest")
plt.show()


# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
