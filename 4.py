import pandas as pd
import matplotlib.pyplot as plt

# Sample feature importance data
feature_importance_data = {
    'Feature': [9, 17, 8, 31, 0, 15, 33, 41, 25, 23, 16, 13, 21, 40, 34, 37, 29, 39, 7, 24, 27, 6, 22, 36, 2, 32, 35, 11, 5, 14, 26, 19, 3, 38, 30, 28, 1, 12, 10, 20, 18, 4],
    'Importance': [0.048801, 0.045294, 0.045166, 0.042762, 0.041304, 0.040141, 0.037130, 0.037126, 0.036201, 0.035987, 0.035341, 0.034001, 0.032253, 0.029663, 0.028305, 0.027140, 0.026074, 0.025394, 0.023224, 0.022543, 0.022096, 0.021486, 0.021264, 0.020973, 0.019538, 0.018783, 0.017818, 0.017317, 0.016100, 0.015495, 0.015255, 0.014867, 0.014222, 0.014188, 0.009688, 0.009534, 0.008488, 0.008138, 0.006387, 0.006137, 0.005082, 0.003297]
}

# Create DataFrame
feature_importance_table = pd.DataFrame(feature_importance_data)

# Reset index
feature_importance_table.reset_index(drop=True, inplace=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_table.index, feature_importance_table['Importance'], color='skyblue')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(feature_importance_table.index, feature_importance_table['Feature'], rotation=90)
plt.tight_layout()

# Save plot as PNG
plt.savefig('feature_importance.png')
plt.show()
