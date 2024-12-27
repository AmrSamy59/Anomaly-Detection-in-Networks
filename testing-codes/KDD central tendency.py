import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Load dataset

# Load the dataset (assuming it's in CSV format)
data = pd.read_csv('kddcup.data_10_percent_corrected', header=None)

# Column 41 (index 41 or -1) is the label
labels = data.iloc[:, -1]

# Count occurrences
label_counts = labels.value_counts()

# Separate normal and attacks
normal_count = label_counts['normal.']
attack_count = label_counts.sum() - normal_count

# Total records
total_count = labels.shape[0]

# Calculate percentages
normal_percentage = (normal_count / total_count) * 100
attack_percentage = (attack_count / total_count) * 100

print(f"Normal connections: {normal_percentage:.2f}%")
print(f"Attack connections: {attack_percentage:.2f}%")

columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
           'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 
           'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
           'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
           'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
           'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
data.columns = columns

# dummy variables
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal.' else 1)
data = pd.get_dummies(data, columns=['protocol_type', 'service', 'flag'])
data = data.dropna()


# drop columns with only one unique value, and columns with string values
data = data.loc[:, data.apply(pd.Series.nunique) != 1]
data = data.select_dtypes(include=[np.number])


statistics = data.describe().T
statistics = statistics.rename(columns={
    'mean': 'Mean',
    'std': 'Standard Deviation',
    '50%': 'Median',
    'max': 'Maximum'
})

def plot_stat(stat_name, log_scale=False):
    plt.figure(figsize=(12, 6))
    data_to_plot = statistics[stat_name]
    sns.barplot(x=statistics.index, y=data_to_plot, palette='viridis', hue=statistics.index, legend=False)
    plt.xticks(rotation=90)
    scale = 'KDD Cup 1999 Dataset\n'
    plt.title(f'{scale}{stat_name} of Features')
    plt.xlabel('Features')
    plt.ylabel(f'{scale}{stat_name}')
    if log_scale:
        plt.yscale('log')
    plt.tight_layout()
    plt.show()

# Plot mean with log transformation
plot_stat('Mean', log_scale=True)

# Plot standard deviation with log transformation
plot_stat('Standard Deviation', log_scale=True)

# Plot median with log transformation
plot_stat('Median', log_scale=True)

# Plot maximum with log transformation
plot_stat('Maximum', log_scale=True)


# Correlation with the target label
# remove columns with similar values variance
correlation_with_label = data.corr()['label'].drop('label').sort_values(ascending=False)
print(correlation_with_label)


# correlations with values > 0.1
correlation_with_label = correlation_with_label[(correlation_with_label > 0.1) | (correlation_with_label < -0.1)]
sns.barplot(x=correlation_with_label.values, y=correlation_with_label.index, hue=correlation_with_label.index, palette='coolwarm', legend=False)
plt.title('Correlation Matrix (for features with correlation > 0.1) with the target label (normal/attack)')
plt.show()