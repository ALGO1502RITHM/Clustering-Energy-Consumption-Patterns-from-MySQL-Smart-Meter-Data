# --- KMenans Model ---

import pandas as pd
import mysql.connector
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---Load data from MySQL into DataFrame---
connection = mysql.connector.connect(
    host="localhost",
    user="energy_user_",
    password="algo1502rithm",
    database="energy_consumption_database"
)

query = "SELECT * FROM energy_consumption"

energy_df = pd.read_sql(query, connection)
connection.close()

# print(energy_df.head())
# print(energy_df.info())

# ---data preprocessing---
time_mappping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
energy_df['time_of_day'] = energy_df['time_of_day'].map(time_mappping).astype('int') # time_of_day converted to category
# print(energy_df['time_of_day'].info())

# --- cyclical features of hour ---
energy_df['hour_sin'] = np.sin(2 * np.pi * energy_df['hour'] / 24)
energy_df['hour_cos'] = np.cos(2 * np.pi * energy_df['hour'] / 24)

# --- Handle missing values--- 
energy_df = energy_df.dropna()

# -- sample data ---
# sample_energy_df = energy_df.sample(n=5000, random_state=42)
# print(sample_energy_df.info())
# print("Sample_energy_df shape:",sample_energy_df.shape)

# --- features ---
features_columns = ['global_active_power', 
                    'total_submetering', 
                    'submetering_ratio', 
                    'kitchen_ratio', 
                    'ac_ratio', 
                    'time_of_day', 
                    'hour_sin', 
                     'hour_cos']

num_features = energy_df[features_columns]
print(num_features.head())
print(num_features.info())
# print("num_features shape:",num_features.shape)

# ---standardizescaler num_features---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(num_features)
print(scaled_features)

# --- Applying PCA ---
pca = PCA(n_components=0.95)
pca_features = pd.DataFrame(pca.fit_transform(scaled_features))
print(pca_features.head())

#  ---Determine number of cluster using silhouette score and elbow method---
intertias = []
silhouette_scores = []
max_k = 10
k_range = range(2, max_k + 1)

for  k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_features)
    intertias.append(kmeans.inertia_)

    if k > 1:
        score = silhouette_score(pca_features, kmeans.labels_, random_state=42)
        silhouette_scores.append(score)
        print(f"k={k}: Silhouette score = {score:.4f}")

#  --- visualization of Elbow and Silhouette score
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, intertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow method') # Elbow visualization

plt.subplot(1, 2, 2)
plt.plot(range(2, max_k + 1), silhouette_scores, 'go-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score') # Silhouette score

plt.tight_layout()
plt.show()

# --- Selecting Optimal K using silhouette score ---
optimal_k = np.argmax(silhouette_scores) + 2 # because k starts at 2
print(f"Optimal number of clusters: {optimal_k}")

# --- Apply cluster using optimal_k [8] ---
Kmeans_model = MiniBatchKMeans(n_clusters= optimal_k, random_state=42, n_init=5, batch_size=1000)
clusters = Kmeans_model.fit_predict(pca_features)

#  --- Add cluster to energy_df ---
energy_df['cluster'] = clusters

#  --- cluster analysis ---
cluster_summary= energy_df.groupby('cluster').agg({
    'global_active_power': 'median',
    'total_submetering': 'median',
    'submetering_ratio': 'median',
    'kitchen_ratio': 'median',
    'ac_ratio': 'median',
    'time_of_day': lambda x: x.mode().iloc[0],
    'hour_sin': 'mean',
    'hour_cos': 'mean'
})
print(cluster_summary)

# --- plot cluster characteristics ---
plt.figure(figsize=(14, 10))
for i, features in enumerate(features_columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='cluster', y=features, data=energy_df)
    plt.title(features)
plt.tight_layout()
plt.show()

# --- PCA visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(pca_features.iloc[:, 0], pca_features.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Cluster visualization (PCA)')
plt.xlabel(f'PCA ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PCA ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(label='Cluster')
plt.show()

# --- Temporal patterns ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='hour', hue='cluster', data=energy_df, palette='viridis')
plt.title('Cluster Distribution by Hour')

plt.subplot(1, 2, 2)
sns.countplot(x='is_weekend', hue='cluster', data=energy_df, palette='magma')
plt.title('Cluster Distribution by is_weekend')
plt.show()

# ---creating cluster labels ---
cluster_centers = pca.inverse_transform(Kmeans_model.cluster_centers_)    # Reverse PCA
original_space_centroids = scaler.inverse_transform(cluster_centers)      # Reverse scaling
centroid_dataframe = pd.DataFrame(original_space_centroids, columns=features_columns)
print(centroid_dataframe)

# --- Calculate relative values ---
overall_median = num_features.median()

# --- Threshold Values ---
High_Value_Threshold = 1.3
Low_Value_Threshold = 0.7

# --- Reverse time mapping ---
reverse_time_mapping = {0: 'Morning', 1: 'Afternoon', 2: 'Evening', 3: 'Night'}

# --- Auto-label Generation Function ---
def auto_label_cluster(row):
    # TIme component
    time_val = int(round(row['time_of_day']))
    time_label = reverse_time_mapping.get(time_val, f"Time{time_val}")

    # Intensity Component
    power_ratio = row['global_active_power'] / overall_median['global_active_power']
    if power_ratio > High_Value_Threshold:
        intensity = "HIGH"
    elif power_ratio < Low_Value_Threshold:
        intensity = "LOW"
    else:
        intensity = "MEDIUM"

    # Appliance Component
    appliance_dominance = []
    if row['kitchen_ratio'] > 0.4 or row['ac_ratio'] > 0.4:
        if row['kitchen_ratio'] > 0.4:
            appliance_dominance.append("Kitchen")
        if row['ac_ratio'] > 0.4:
            appliance_dominance.append('AC')
    
    if not appliance_dominance:
        if row['submetering_ratio'] < 0.25:
            appliance = "Non_Metered Heavy"
        else:
            appliance = "Balanced"
    else:
        appliance = f"{'&'.join(appliance_dominance)}-Heavy"

    return f"{intensity} {time_label} {appliance}".strip()

# Apply to Centriods
centroid_dataframe['Cluster_Label'] = centroid_dataframe.apply(auto_label_cluster, axis=1)
print("\nCentroid DataFrame:", centroid_dataframe)

# --- Apply to sample dataframe ---
label_mapping = centroid_dataframe['Cluster_Label'].to_dict()
energy_df['Cluster_Label'] = energy_df['cluster'].map(label_mapping)
print("\nEnergy DataFrame including Cluster Label:", energy_df.head())

# --- Visaulize labels ---
plt.figure(figsize=(12, 8))
sns.countplot(y='Cluster_Label', data=energy_df, order=energy_df['Cluster_Label'].value_counts().index)
plt.title('Cluster Label Distribution')
plt.xlabel('Count')
plt.ylabel('Cluster Label')
plt.show()

# --- label Patterns ---
print("\nCluster Label Pattern:")
print(energy_df.groupby('Cluster_Label').agg({
    'global_active_power': ['median', 'mean'],
    'hour': lambda x: x.mode()[0],
    'is_weekend': 'mean'
}))

# --- Hourly Label Pattern ---
plt.figure(figsize=(14, 10))
for i, label in enumerate(energy_df['Cluster_Label'].unique()):
    plt.subplot(5, 2, i+1)
    label_data = energy_df[energy_df['Cluster_Label'] == label]
    sns.lineplot(data=label_data, x='hour', y='global_active_power', errorbar=None)
    plt.title(label)
    plt.ylim(0, energy_df['global_active_power'].max()*1.1)
plt.tight_layout()
plt.show()

