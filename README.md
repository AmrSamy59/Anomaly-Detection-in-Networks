# Anomaly Detection in Network Traffic for Security

## Overview
This project, **"Anomaly Detection in Network Traffic for Security"**, focuses on identifying and categorizing malicious traffic in network systems to enhance cybersecurity. By leveraging statistical analysis and machine learning models, the system can detect anomalies and classify various attack types, providing a robust solution for mitigating cyber threats.

---

## Key Features
- **Real-Time Detection:** Identifies unusual patterns in network traffic that may indicate security threats such as denial-of-service (DoS) attacks, data breaches, or unauthorized access.
- **Attack Categorization:** Differentiates between normal traffic and multiple types of attacks, including DoS, probing, R2L (Remote-to-Local), and U2R (User-to-Root).
- **Advanced Analytics:** Utilizes datasets like KDD Cup 1999 and UNSW-NB15 to train models and validate results.
- **Machine Learning Integration:** Employs algorithms like Random Forest and Isolation Forest for accurate anomaly classification.

---

## Datasets Used
### 1. **KDD Cup 1999**
- Contains 41 features and over 4.9 million records.
- Attack types include DoS, probing, R2L, and U2R.

### 2. **UNSW-NB15**
- Features modern attack scenarios with 49 attributes and over 2.5 million records.
- Attack categories include fuzzers, DoS, reconnaissance, backdoors, and worms.

---

## Methods and Techniques
### Statistical Analysis
- **ANOVA:** Identifies significant differences in feature values across attack categories.
- **Chi-Square Test:** Evaluates associations between categorical features and anomaly labels.

### Machine Learning Models
- **Random Forest:** Achieved accuracy of 99.98% (KDD) and 98.31% (UNSW).
- **Isolation Forest:** Detects anomalies by isolating outliers efficiently.

---

## Results
### KDD Cup 1999 Dataset
- **ANOVA Result:** Significant F-statistic (25.93) with p-value (1.5487e-106).
- **Random Forest Accuracy:** 99.98%.
- **Isolation Forest Anomalies Detected:** 14,810.

### UNSW-NB15 Dataset
- **ANOVA Result:** Significant F-statistic (96.78) with p-value (7.83e-23).
- **Random Forest Accuracy:** 98.31%.
- **Isolation Forest Anomalies Detected:** 7,763.

---

## Team Members
- [Amr Samy](https://github.com/AmrSamy59) – Team Leader
- [Ali Eldien Alaa](https://github.com/AliAlaa88) – Member
- [Abdullah Ayman](https://github.com/AbdallahAyman03) – Member
- [Seif Eldien Mohamed](https://github.com/im-saif) – Member
- [Esraa Hassan](https://github.com/Esraa-Hassan0) – Member
- [Nawal Hossam](https://github.com/Nawalhossam) – Member

---

## Future Work
- Real-time deployment and testing in live network environments.
- Exploring unsupervised learning techniques for enhanced anomaly detection.

---

