# # -- K-Prototypes --

# import pandas as pd
# import mysql.connector
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from kmodes.kprototypes import KPrototypes
# from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
# from sklearn.ensemble import IsolationForest
# from scipy.stats import median_abs_deviation
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
# import gower

# # -- sample configuration --
# sample_size = 5000           # sample data representation
# k_range = range(2, 11)       # test 2-10 clusters
# RANDOM_STATE = 42
# n_jobs = -1                  # uses all available cores
# OUTLIER_CAPPING = 0.05       

# # -- Outlier handling --
# def handle_outliers(df, numerical_features):
#     """Comprehension outlier management pipeline"""
#     df_clean = df.copy()

#     # IQR outlier handler for each numerical feature
#     for feature in numerical_features:
#         q05 = df[feature].quantile(0.05)
#         q95 = df[feature].quantile(0.95)
#         iqr = q95 - q05

#         # wider bounds for energy data
#         lower_bound = q05 - 3 * iqr
#         upper_bound = q95 + 3 * iqr

#         # cap outliers
#         df_clean[feature] = df[feature].clip(lower_bound, upper_bound)

#     # isolation forest for multivariate outliers
#     iso_forest = IsolationForest(
#         contamination=OUTLIER_CAPPING,
#         random_state=RANDOM_STATE,
#         n_jobs=n_jobs
#     )
#     # fit only on numerical_features
#     outliers = iso_forest.fit_predict(df_clean[numerical_features])
#     df_clean['outlier_flag'] = outliers == -1

#     return df_clean

# # -- robust validation metrics --
# def robust_validation(data, clusters, numerical_features):
#     """Cluster vallidation metrics resistant to outliers"""
#     # median based silhouette 
#     from sklearn.metrics import pairwise_distances
#     from sklearn.metrics import silhouette_samples

#     # use manhattan distance instead euclidean distance for robustness
#     dist_matrix = pairwise_distances(data[numerical_features], metric='manhattan')
#     sil_samples = silhouette_samples(dist_matrix, clusters, metric='precomputed')

#     # use median intead of mean for robustness
#     robust_silhouette = np.median(sil_samples)

#     # Calinski-Harabasz with MAD-scaled data
#     mad_vals = median_abs_deviation(data[numerical_features], axis=0, scale='normal')
#     scaled_data = data[numerical_features] / mad_vals
#     ch_score = calinski_harabasz_score(scaled_data, clusters)

#     return{
#         'robust_silhouette': robust_silhouette,
#         'robust_calinski_harabasz': ch_score
#     }


# # -- load data from MySQL --
# connection = mysql.connector.connect(
#     host="localhost",
#     user="energy_user_",
#     password="algo1502rithm",
#     database="energy_consumption_database"
# )
# query = "SELECT * FROM energy_consumption"

# energy_df = pd.read_sql(query, connection)
# connection.close()
# print(energy_df.head())

# # -- Data Preprocessing --
# energy_df['time_of_day'] = energy_df['time_of_day'].astype(str).str.strip()
# time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
# energy_df['time_of_day'] = energy_df['time_of_day'].map(time_mapping)

# # --- cyclical features of hour ---
# energy_df['hour_sin'] = np.sin(2 * np.pi * energy_df['hour'] / 24)
# energy_df['hour_cos'] = np.cos(2 * np.pi * energy_df['hour'] / 24)

# # -- is_weekend to categorical --
# energy_df['is_weekend'] = energy_df['is_weekend'].astype('int')

# # -- Handling Missing Values --
# energy_df = energy_df.dropna().copy()
# print(energy_df.head())

# # -- Seperate numerical and categorical features --
# numerical_features = ['global_active_power', 'total_submetering', 'submetering_ratio', 'kitchen_ratio', 'ac_ratio','hour_sin', 'hour_cos']  # numerical features
# categorical_features = ['is_weekend', 'time_of_day']

# # -- applying outlier handling --
# print("\nApplying outlier handling: ")
# energy_df = handle_outliers(energy_df, numerical_features)
# main_data = energy_df[~energy_df['outlier_flag']]
# outlier_data = energy_df[energy_df['outlier_flag']]

# print(f"Main data: {len(main_data)} records ({len(main_data)/len(energy_df):.1%})")
# print(f"Outliers data: {len(outlier_data)} records ({len(outlier_data)/len(energy_df):.1%})")

# # Outlier handling visualization distribution
# plt.figure(figsize=(12, 6))
# sns.scatterplot(x='hour', y='global_active_power', data=energy_df, hue='outlier_flag',
#                 palette={True: 'red', False: 'blue'}, alpha=0.6)
# plt.title('Hour outlier visualization distribution')
# plt.tight_layout()
# plt.show()

# # --- Preparing Main data for clustering 
# # scale numerical features using robustscaler
# medians = main_data[numerical_features].median()
# mads = median_abs_deviation(main_data[numerical_features], axis=0, scale='normal')
# main_data_scaled = main_data.copy()
# main_data_scaled[numerical_features] = (main_data[numerical_features] - medians) / mads

# # validation sample 
# validation_sample = main_data_scaled.sample(sample_size, random_state=RANDOM_STATE)

# # preparing columns with categorical data at the end
# Kproto_data = validation_sample[numerical_features + categorical_features].copy()
# for col in categorical_features:
#     Kproto_data[col] = Kproto_data[col].astype(str)

# # -- Indices of categorical_features column (last 2 columns)
# categorical_indices = [Kproto_data.columns.get_loc(col) for col in categorical_features]

# # -- create bolean mask categorical features
# cat_mask = [col in categorical_features for  col in Kproto_data.columns]

# # -- cluster validation --
# validation_result = []

# print('Starting clustering...')
# print('Sample size:', sample_size)
# print('Testing k-values:', k_range)

# for k in k_range:
#     iteration_start = time.time()

#     print(f"\n--- Testing k={k} ---")

#     # Train K-Prototypes
#     kproto = KPrototypes(
#         n_clusters=k,
#         init='Cao',
#         verbose=0,
#         random_state=RANDOM_STATE,
#         n_jobs=n_jobs
#     )

#     try:
#         clusters = kproto.fit_predict(Kproto_data, categorical=categorical_indices)

#         # compute metrics
#         metrics={
#             'k': k,
#             'cost': kproto.cost_,
#             'time': time.time() - iteration_start
#         }

#         # numerical metrics
#         numerical_data = Kproto_data.iloc[:, :len(numerical_features)]
#         metrics['davies_bouldin'] = davies_bouldin_score(numerical_data, clusters)
#         metrics['calinski_harabasz'] = calinski_harabasz_score(numerical_data, clusters)

#         # robust_metrics
#         robust_metrics = robust_validation(Kproto_data, clusters, numerical_features)
#         metrics.update(robust_metrics)

#         # Gower silhouette
#         try:
#             gower_dist = gower.gower_matrix(Kproto_data, cat_features=cat_mask)
#             metrics['gower_silhouette'] = silhouette_score(gower_dist, clusters, metric='precomputed')

#         except Exception as e:
#             print(f"Gower-Silhouettte failed: {str(e)}")
#             metrics['gower_silhouette'] = None

#         validation_result.append(metrics)

#         print(f" Cost: {metrics['cost']:.0f}")
#         print(f" Robust Silhouette: {metrics['robust_silhouette']:.4f}")
#         print(f" Robust CH: {metrics['robust_calinski_harabasz']:.0f}")
#         if metrics['gower_silhouette'] is not None:
#             print(f"Gower Silhouette: {metrics['gower_silhouette']}")
#         print(f" Time: {metrics['time']:.1f}s")

#     except Exception as e:
#         print(f" clustering failed for k={k}: {str(e)}")
#         validation_result.append({
#             'k': k,
#              'error': str(e)
#         })

# # converts to dataframe
# results_dataframe = pd.DataFrame(validation_result)
# print(results_dataframe.head())

# # --- Determine Optimal k ---
# # use robust silhouette as primary metric
# if 'robust_silhouette' in results_dataframe and results_dataframe['robust_silhouette'].notnull().any():
#     optimal_k = int(results_dataframe.loc[results_dataframe['robust_silhouette'].idxmax()]['k'])
#     print(f"Optimal_k Using robust silhouette: {optimal_k}")

# else:
#     # falllback to cost elbow method
#     costs = results_dataframe['cost'].values
#     k_values = results_dataframe['k'].values
#     first_diff = -np.diff(costs)
#     second_diff = np.diff(first_diff)
#     elbow_index = np.argmax(second_diff) + 2
#     optimal_k = int(k_values[elbow_index])
#     print(f"\nOptimal k by elbow method: {optimal_k}")

# # --- visualization of validation metrics ---
# print("\n--- visualization of validation metrics ---")

# plt.figure(figsize=(15, 10))

# # cost function
# plt.subplot(2, 2, 1)
# plt.plot(results_dataframe['k'], results_dataframe['cost'], 'bo-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Costs')
# plt.title('K-Prototypes Cost Function')
# plt.grid(True)

# # Robust Silhouettte
# plt.subplot(2, 2, 2)
# plt.plot(results_dataframe['k'], results_dataframe['robust_silhouette'], 'mo-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Robust Silhouette score')
# plt.title('K-Prototypes Robust silhouette')
# plt.grid(True)

# # Robust Calinski Harabasz
# plt.subplot(2, 2, 3)
# plt.plot(results_dataframe['k'], results_dataframe['robust_calinski_harabasz'], 'go-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Robust Calinski score')
# plt.title('K-Prototypes Robust Calinski Harabasz')
# plt.grid(True)

# # Gower Silhouette
# if 'gower_silhouette' in results_dataframe:
#     plt.subplot(2, 2, 4)
#     plt.plot(results_dataframe['k'], results_dataframe['gower_silhouette'], 'co-')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Gower Silhouette score')
#     plt.title('Mixed-Type Cluster Quality')
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- Final Model Training ---
# print(f"\nTraining final model using {optimal_k}")

# # load full clean dataset
# full_clean_data = main_data_scaled[numerical_features + categorical_features].copy()
# for col in categorical_features:
#     full_clean_data[col] = full_clean_data[col].astype(str)

# # recompute categorical indicies
# recompute_categorical_indices = [full_clean_data.columns.get_loc(col) for col in categorical_features]

# # Final model Train
# final_kproto = KPrototypes(
#     n_clusters = optimal_k,
#     init = 'Cao',
#     verbose = 1,
#     random_state = RANDOM_STATE,
#     n_jobs = n_jobs
# )

# start_time = time.time()
# final_model_clean_clusters = final_kproto.fit_predict(full_clean_data, categorical=recompute_categorical_indices)
# training_time = time.time() - start_time

# print(f"Fianl Training Time completed in {training_time:.1f} seconds")

# # --- assign clusters ---
# main_data['cluster'] =  final_model_clean_clusters
# outlier_data.loc[:, 'cluster'] = -1                       # assign outliers to special characters

# # combining results
# final_dataframe = pd.concat([main_data, outlier_data])

# # --- Cluster Analysis ---
# # Reverse scaling for interpretation
# for i, col in enumerate(numerical_features):
#     mask = final_dataframe['cluster'] != -1
#     final_dataframe.loc[mask, col] = final_dataframe.loc[mask, col] * mads[i] + medians[col]

# # Create cluster lables function
# print("\n Auto Creating Cluster label")

# def auto_create_clusters_label(cluster_id):
#     if cluster_id == -1:
#         return 'Outlier'
    
#     cluster_data = final_dataframe[final_dataframe['cluster'] == cluster_id].copy()

#     # intensity
#     power_median = cluster_data['global_active_power'].median()
#     if power_median < 0.5:
#         intensity = "Low"
#     elif power_median < 1.0:
#         intensity = "Medium"
#     else:
#         intensity = "High"

#     # Time of day

#     print(cluster_data['time_of_day'].unique())
#     print(type(cluster_data['time_of_day'].iloc[0]))

#     # convert time_of_day numeric incase it's stilll string from clustering
#     cluster_data['time_of_day'] = pd.to_numeric(cluster_data['time_of_day'], errors='coerce')
#     print(f"Cluster {cluster_id} time_of_day unique values: {cluster_data['time_of_day'].unique()}")

#     # calculate mode, handling multiple mode or null 
#     time_mode = cluster_data['time_of_day'].mode(dropna=True)

#     if time_mode.empty and pd.notna(time_mode.iloc[0]):
#         time_label = 'Unknown'
#     else:
#         try:
#             time_mode_val = int(float(time_mode.iloc[0]))
#             # reverse time_mapping
#             reverse_time_mapping = {0: 'Morning', 1: 'Afternoon', 2: 'Evening', 3: 'Night'}
#             time_label = reverse_time_mapping.get(time_mode_val, f"Invalid ({time_mode_val})")
                
#         except (ValueError, IndexError) as e:
#                 print(f"Error processing time_of_day mode for cluster {cluster_id}: {str(e)}")
#                 time_label = 'Unknown'

#     # Appliance Dominated
#     kitchen_median = cluster_data['kitchen_ratio'].median()
#     ac_median = cluster_data['ac_ratio'].median()

#     if kitchen_median > 0.4 and ac_median > 0.4:
#         appliance = "Kitchen+Ac Dominated"
#     elif kitchen_median > 0.4:
#         appliance = "Kitchen Dominated"
#     elif ac_median > 0.4:
#         appliance = "Ac Dominated"
#     else:
#         appliance = "Balanced"

#     # weekend pattern (is_weekend)
#     weekend_prop = cluster_data['is_weekend'].mean()
#     if weekend_prop > 0.6:
#         temporal = "Weekend"
#     elif weekend_prop < 0.4:
#         temporal = "Weekday"
#     else:
#         temporal = "Mixed"

#     return f"{intensity} {time_label} {appliance} {temporal}"

# # Apply auto create clusters label
# cluster_labels = {}
# for cluster_id in final_dataframe['cluster'].unique():
#     cluster_labels[cluster_id] = auto_create_clusters_label(cluster_id)

# final_dataframe['cluster_labels'] = final_dataframe['cluster'].map(cluster_labels)
# print(final_dataframe.head())

# # --- Visaulize cluster results ---
# print("\nVisualizating Cluster Results")

# plt.figure(figsize=(14, 8))

# # cluster distribution
# plt.subplot(1, 2, 1)
# sns.countplot(x='cluster_labels', data=final_dataframe, order=final_dataframe['cluster_labels'].value_counts().index)
# plt.title('Cluster Distribution')
# plt.xticks(rotation=90)

# # Hourly Patterns
# plt.subplot(1, 2, 2)
# for label in final_dataframe['cluster_labels'].unique():
#     if label == "Outlier":
#         continue
#     label_data = final_dataframe[final_dataframe['cluster_labels'] == label]
#     hourly_avg = label_data.groupby('hour')['global_active_power'].median()
#     plt.plot(hourly_avg, label=label)

# plt.title('Hourly Consumption pattern')
# plt.xlabel('Hour of Day')
# plt.ylabel('Medain Global Active Power')
# plt.legend()
# plt.tight_layout()
# plt.show

# # --- Outliers Results ---
# if len(outlier_data) > 0:
#     print('\nOutlier Analysis')
#     outlier_summary = outlier_data.groupby('hour')['global_active_power'].agg(['count', 'median'])
#     print(outlier_summary)

#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='hour', y='global_active_power', data=outlier_data, alpha=0.7)
#     plt.title('Outlier consumption by Hour')
#     plt.show()



# -- K-Prototypes --

import pandas as pd
import mysql.connector
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.ensemble import IsolationForest
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gower

# -- sample configuration --
sample_size = 5000  # sample data representation
k_range = range(2, 11)  # test 2-10 clusters
RANDOM_STATE = 42
n_jobs = -1  # uses all available cores
OUTLIER_CAPPING = 0.05

# -- Outlier handling --
def handle_outliers(df, numerical_features):
    """Comprehensive outlier management pipeline"""
    df_clean = df.copy()

    # IQR outlier handler for each numerical feature
    for feature in numerical_features:
        # Use 1.5 * IQR for standard outlier detection, or adjust as needed
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Wider bounds for energy data, as per your original logic
        lower_bound = df[feature].quantile(0.05) - 3 * IQR # Use original 0.05 quantile, but still multiply by IQR
        upper_bound = df[feature].quantile(0.95) + 3 * IQR # Use original 0.95 quantile, but still multiply by IQR


        # Cap outliers
        df_clean[feature] = df[feature].clip(lower_bound, upper_bound)

    # Isolation Forest for multivariate outliers
    iso_forest = IsolationForest(
        contamination=OUTLIER_CAPPING,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs
    )
    # Fit only on numerical_features
    outliers = iso_forest.fit_predict(df_clean[numerical_features])
    df_clean['outlier_flag'] = outliers == -1

    return df_clean

# -- Robust Validation Metrics --
def robust_validation(data, clusters, numerical_features):
    """Cluster validation metrics resistant to outliers"""
    from sklearn.metrics import pairwise_distances
    from sklearn.metrics import silhouette_samples

    # Ensure data[numerical_features] is a DataFrame or numpy array
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Use Manhattan distance instead of Euclidean distance for robustness
    # Ensure all numerical features are present in the data being passed
    dist_matrix = pairwise_distances(data[numerical_features], metric='manhattan')
    
    # Check if there's more than one cluster to compute silhouette score
    if len(np.unique(clusters)) > 1:
        sil_samples = silhouette_samples(dist_matrix, clusters, metric='precomputed')
        # Use median instead of mean for robustness
        robust_silhouette = np.median(sil_samples)
    else:
        robust_silhouette = np.nan # Cannot compute silhouette for a single cluster

    # Calinski-Harabasz with MAD-scaled data
    # Ensure MAD values are not zero to avoid division by zero
    mad_vals = median_abs_deviation(data[numerical_features], axis=0, scale='normal')
    mad_vals[mad_vals == 0] = 1e-9 # Replace zero MADs with a small number to prevent division by zero

    scaled_data = data[numerical_features] / mad_vals
    
    # Check if there's more than one cluster to compute Calinski-Harabasz
    if len(np.unique(clusters)) > 1:
        ch_score = calinski_harabasz_score(scaled_data, clusters)
    else:
        ch_score = np.nan # Cannot compute Calinski-Harabasz for a single cluster

    return {
        'robust_silhouette': robust_silhouette,
        'robust_calinski_harabasz': ch_score
    }


# -- Load data from MySQL --
# It's good practice to use a try-except block for database connections
connection = None # Initialize connection to None
try:
    connection = mysql.connector.connect(
        host="localhost",
        user="energy_user_",
        password="algo1502rithm",
        database="energy_consumption_database"
    )
    query = "SELECT * FROM energy_consumption"
    energy_df = pd.read_sql(query, connection)
    print("Successfully loaded data from MySQL.")
    print(energy_df.head())
except mysql.connector.Error as err:
    print(f"Error connecting to MySQL or executing query: {err}")
    # Exit or handle the error appropriately if data loading is critical
    import sys
    sys.exit(1) # Exit the script if database connection fails
finally:
    if connection and connection.is_connected():
        connection.close()
        print("MySQL connection closed.")

# -- Data Preprocessing --
# Ensure 'time_of_day' column exists before processing
if 'time_of_day' in energy_df.columns:
    energy_df['time_of_day'] = energy_df['time_of_day'].astype(str).str.strip()
    time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    energy_df['time_of_day'] = energy_df['time_of_day'].map(time_mapping).fillna(-1).astype(int) # Handle potential NaNs after mapping
else:
    print("Warning: 'time_of_day' column not found. Skipping time_of_day processing.")

# --- Cyclical features of hour ---
# Ensure 'hour' column exists
if 'hour' in energy_df.columns:
    energy_df['hour_sin'] = np.sin(2 * np.pi * energy_df['hour'] / 24)
    energy_df['hour_cos'] = np.cos(2 * np.pi * energy_df['hour'] / 24)
else:
    print("Warning: 'hour' column not found. Skipping cyclical hour features.")

# -- is_weekend to categorical --
if 'is_weekend' in energy_df.columns:
    energy_df['is_weekend'] = energy_df['is_weekend'].astype('int')
else:
    print("Warning: 'is_weekend' column not found. Skipping is_weekend conversion.")


# -- Handling Missing Values --
# It's better to dropna early if missing values are not to be imputed.
# Use .copy() after operations that might return a view to avoid SettingWithCopyWarning later.
energy_df = energy_df.dropna().copy()
print("\nAfter dropping missing values:")
print(energy_df.head())

# -- Separate numerical and categorical features --
# Make sure these columns actually exist in your DataFrame after preprocessing
numerical_features = [
    'global_active_power', 'total_submetering', 'submetering_ratio',
    'kitchen_ratio', 'ac_ratio', 'hour_sin', 'hour_cos'
]
categorical_features = ['is_weekend', 'time_of_day']

# Filter features to only include those present in the DataFrame
numerical_features = [f for f in numerical_features if f in energy_df.columns]
categorical_features = [f for f in categorical_features if f in energy_df.columns]

if not numerical_features:
    print("Error: No numerical features found after preprocessing. Exiting.")
    sys.exit(1)
if not categorical_features:
    print("Error: No categorical features found after preprocessing. Exiting.")
    sys.exit(1)

# -- Applying outlier handling --
print("\nApplying outlier handling:")
energy_df = handle_outliers(energy_df, numerical_features)
main_data = energy_df[~energy_df['outlier_flag']].copy() # Use .copy() to avoid SettingWithCopyWarning
outlier_data = energy_df[energy_df['outlier_flag']].copy() # Use .copy()

print(f"Main data: {len(main_data)} records ({len(main_data)/len(energy_df):.1%})")
print(f"Outliers data: {len(outlier_data)} records ({len(outlier_data)/len(energy_df):.1%})")

# Outlier handling visualization distribution
plt.figure(figsize=(12, 6))
# Ensure 'hour' and 'global_active_power' are available for plotting
if 'hour' in energy_df.columns and 'global_active_power' in energy_df.columns:
    sns.scatterplot(x='hour', y='global_active_power', data=energy_df, hue='outlier_flag',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.title('Hour vs. Global Active Power with Outlier Flag')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping outlier visualization: 'hour' or 'global_active_power' column not found.")

# --- Preparing Main data for clustering
# Scale numerical features using robustscaler (manual implementation with median and MAD)
if not main_data.empty:
    medians = main_data[numerical_features].median()
    mads = median_abs_deviation(main_data[numerical_features], axis=0, scale='normal')
    
    # Handle cases where MAD might be zero (e.g., constant feature)
    mads[mads == 0] = 1e-9 # Replace zero MADs with a small number to prevent division by zero

    main_data_scaled = main_data.copy()
    main_data_scaled[numerical_features] = (main_data[numerical_features] - medians) / mads
else:
    print("Warning: main_data is empty after outlier handling. Cannot proceed with clustering.")
    sys.exit(0) # Exit gracefully if no data to cluster

# Validation sample
if len(main_data_scaled) >= sample_size:
    validation_sample = main_data_scaled.sample(sample_size, random_state=RANDOM_STATE).copy()
else:
    print(f"Warning: Sample size ({sample_size}) is larger than available main data ({len(main_data_scaled)}). Using all available data for validation.")
    validation_sample = main_data_scaled.copy()


# Preparing columns with categorical data at the end
# Ensure that all specified numerical and categorical features exist in validation_sample
Kproto_data_cols = [col for col in numerical_features + categorical_features if col in validation_sample.columns]
Kproto_data = validation_sample[Kproto_data_cols].copy()

# Ensure categorical columns are treated as strings for KPrototypes
for col in categorical_features:
    if col in Kproto_data.columns:
        Kproto_data[col] = Kproto_data[col].astype(str)

# -- Indices of categorical_features column
# Get indices dynamically for categorical features within Kproto_data
categorical_indices = [Kproto_data.columns.get_loc(col) for col in categorical_features if col in Kproto_data.columns]

# -- Create boolean mask for categorical features for gower
cat_mask = [col in categorical_features for col in Kproto_data.columns]


# -- Cluster Validation --
validation_result = []

print('\nStarting clustering...')
print('Sample size:', len(Kproto_data)) # Print actual sample size used
print('Testing k-values:', k_range)

for k in k_range:
    iteration_start = time.time()

    print(f"\n--- Testing k={k} ---")

    # Train K-Prototypes
    kproto = KPrototypes(
        n_clusters=k,
        init='Cao',
        verbose=0,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs
    )

    try:
        # Check if there are enough samples for the current k
        if len(Kproto_data) < k:
            print(f"Skipping k={k}: Not enough samples ({len(Kproto_data)}) for {k} clusters.")
            validation_result.append({
                'k': k,
                'error': 'Not enough samples for this k'
            })
            continue

        clusters = kproto.fit_predict(Kproto_data, categorical=categorical_indices)

        # Compute metrics
        metrics = {
            'k': k,
            'cost': kproto.cost_,
            'time': time.time() - iteration_start
        }

        # Numerical metrics - ensure `numerical_features` are correctly subsetted
        # Ensure that `numerical_data` only contains numerical features used for clustering
        numerical_data_for_metrics = Kproto_data[[f for f in numerical_features if f in Kproto_data.columns]]
        
        # Only compute scores if there's more than one cluster
        if len(np.unique(clusters)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(numerical_data_for_metrics, clusters)
            metrics['calinski_harabasz'] = calinski_harabasz_score(numerical_data_for_metrics, clusters)
        else:
            metrics['davies_bouldin'] = np.nan
            metrics['calinski_harabasz'] = np.nan

        # Robust metrics - pass the relevant data and features
        robust_metrics = robust_validation(Kproto_data, clusters, numerical_features=[f for f in numerical_features if f in Kproto_data.columns])
        metrics.update(robust_metrics)

        # Gower silhouette
        try:
            # Ensure cat_features in gower.gower_matrix matches the indices of categorical columns in Kproto_data
            # gower expects a list of booleans or integer indices for categorical columns.
            # The cat_mask created earlier is a boolean mask for columns in Kproto_data.
            gower_dist = gower.gower_matrix(Kproto_data, cat_features=cat_mask)
            
            # Check if there's more than one cluster to compute silhouette score
            if len(np.unique(clusters)) > 1:
                metrics['gower_silhouette'] = silhouette_score(gower_dist, clusters, metric='precomputed')
            else:
                metrics['gower_silhouette'] = np.nan

        except Exception as e:
            print(f"Gower-Silhouette failed for k={k}: {str(e)}")
            metrics['gower_silhouette'] = None

        validation_result.append(metrics)

        print(f" Cost: {metrics['cost']:.0f}")
        print(f" Robust Silhouette: {metrics['robust_silhouette']:.4f}")
        print(f" Robust CH: {metrics['robust_calinski_harabasz']:.0f}")
        if metrics['gower_silhouette'] is not None:
            print(f"Gower Silhouette: {metrics['gower_silhouette']:.4f}") # Format for consistency
        print(f" Time: {metrics['time']:.1f}s")

    except Exception as e:
        print(f"Clustering failed for k={k}: {str(e)}")
        validation_result.append({
            'k': k,
            'error': str(e),
            'cost': np.nan, # Add NaNs for consistency in DataFrame
            'time': time.time() - iteration_start,
            'robust_silhouette': np.nan,
            'robust_calinski_harabasz': np.nan,
            'gower_silhouette': np.nan
        })

# Convert to dataframe
results_dataframe = pd.DataFrame(validation_result)
print("\nValidation Results:")
print(results_dataframe)

# --- Determine Optimal k ---
optimal_k = None
# Use robust silhouette as primary metric
if 'robust_silhouette' in results_dataframe.columns and results_dataframe['robust_silhouette'].notnull().any():
    # Filter out NaNs for idxmax to work correctly
    temp_df = results_dataframe.dropna(subset=['robust_silhouette'])
    if not temp_df.empty:
        optimal_k = int(temp_df.loc[temp_df['robust_silhouette'].idxmax()]['k'])
        print(f"\nOptimal k Using robust silhouette: {optimal_k}")
    else:
        print("\nRobust silhouette scores are all NaN. Falling back to elbow method.")

if optimal_k is None:
    # Fallback to cost elbow method if robust silhouette is not available or all NaNs
    if 'cost' in results_dataframe.columns and results_dataframe['cost'].notnull().any():
        # Ensure costs are in descending order for diff to make sense
        costs = results_dataframe['cost'].values
        k_values = results_dataframe['k'].values

        # Find the "elbow" point - looking for the point of maximum curvature (largest decrease in the rate of change)
        # This is a heuristic and might need adjustment based on the cost curve shape.
        if len(costs) > 2:
            first_diff = np.diff(costs)
            second_diff = np.diff(first_diff)
            # The elbow is typically where the second derivative is maximized (most negative)
            # or where the absolute value of the second derivative is largest.
            # Here, we're looking for the largest *change* in the decrease.
            # A common heuristic is to find the point where the cost reduction starts to diminish significantly.
            
            # Simple elbow method: find the point where the decrease in cost starts to flatten out.
            # This can be approximated by finding the k where the second derivative is largest.
            # np.argmax(second_diff) gives the index in second_diff.
            # If second_diff has N elements, first_diff has N+1, costs has N+2.
            # So, elbow_index refers to the k_values index.
            
            # More robust elbow detection can be complex. For simplicity, let's look for the max curvature.
            # The indices for second_diff correspond to k_values[2:]
            
            # Consider filtering out NaNs from cost before calculation
            valid_costs_df = results_dataframe.dropna(subset=['cost'])
            if len(valid_costs_df) > 2:
                valid_costs = valid_costs_df['cost'].values
                valid_k_values = valid_costs_df['k'].values
                
                # Calculate the "knee" or "elbow" point using a more robust method if possible,
                # or stick to the simple second derivative for now.
                # The provided elbow calculation is for a specific interpretation of "elbow".
                # Let's use a simpler heuristic for now. The point where the rate of change decreases significantly.
                # This often corresponds to the largest "bend" in the curve.
                
                # A common approach for elbow: find the largest drop in cost relative to the previous drop.
                # Or, using the "knee point" algorithm from kneed or similar.
                # For now, keeping your existing second_diff logic.
                if len(second_diff) > 0:
                    elbow_index_in_second_diff = np.argmax(second_diff)
                    # The k value corresponding to this elbow_index.
                    # If second_diff[i] corresponds to k_values[i+2]
                    optimal_k = int(valid_k_values[elbow_index_in_second_diff + 2])
                    print(f"\nOptimal k by elbow method (second derivative): {optimal_k}")
                else:
                    print("\nNot enough data points to compute elbow method (second derivative).")
                    optimal_k = results_dataframe['k'].min() # Fallback to smallest k
            else:
                print("\nNot enough valid cost values to compute elbow method. Falling back to smallest k.")
                optimal_k = results_dataframe['k'].min() # Fallback to smallest k
        else:
            print("\nNot enough k values to compute elbow method. Falling back to smallest k.")
            optimal_k = results_dataframe['k'].min() # Fallback to smallest k
    else:
        print("\nCost function data is not available or all NaN. Cannot determine optimal k.")
        # As a last resort, pick the smallest k or a default
        optimal_k = k_range[0] if k_range else 2 # Default to 2 if k_range is empty
        print(f"Defaulting to k={optimal_k}")


# Ensure optimal_k is set before proceeding
if optimal_k is None:
    print("Could not determine an optimal k. Exiting.")
    sys.exit(1)


# --- Visualization of Validation Metrics ---
print("\n--- Visualization of Validation Metrics ---")

plt.figure(figsize=(15, 10))

# Cost function
plt.subplot(2, 2, 1)
if 'cost' in results_dataframe.columns and results_dataframe['cost'].notnull().any():
    plt.plot(results_dataframe['k'], results_dataframe['cost'], 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Costs (Inertia)')
    plt.title('K-Prototypes Cost Function')
    plt.grid(True)
else:
    plt.text(0.5, 0.5, 'Cost function data not available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('K-Prototypes Cost Function (N/A)')
plt.xticks(results_dataframe['k'].values) # Set x-ticks to k values

# Robust Silhouette
plt.subplot(2, 2, 2)
if 'robust_silhouette' in results_dataframe.columns and results_dataframe['robust_silhouette'].notnull().any():
    plt.plot(results_dataframe['k'], results_dataframe['robust_silhouette'], 'mo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Robust Silhouette Score')
    plt.title('K-Prototypes Robust Silhouette')
    plt.grid(True)
else:
    plt.text(0.5, 0.5, 'Robust Silhouette data not available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('K-Prototypes Robust Silhouette (N/A)')
plt.xticks(results_dataframe['k'].values)

# Robust Calinski Harabasz
plt.subplot(2, 2, 3)
if 'robust_calinski_harabasz' in results_dataframe.columns and results_dataframe['robust_calinski_harabasz'].notnull().any():
    plt.plot(results_dataframe['k'], results_dataframe['robust_calinski_harabasz'], 'go-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Robust Calinski-Harabasz Score')
    plt.title('K-Prototypes Robust Calinski-Harabasz')
    plt.grid(True)
else:
    plt.text(0.5, 0.5, 'Robust Calinski-Harabasz data not available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('K-Prototypes Robust Calinski-Harabasz (N/A)')
plt.xticks(results_dataframe['k'].values)

# Gower Silhouette
plt.subplot(2, 2, 4)
if 'gower_silhouette' in results_dataframe.columns and results_dataframe['gower_silhouette'].notnull().any():
    plt.plot(results_dataframe['k'], results_dataframe['gower_silhouette'], 'co-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Gower Silhouette Score')
    plt.title('Mixed-Type Cluster Quality (Gower Silhouette)')
    plt.grid(True)
else:
    plt.text(0.5, 0.5, 'Gower Silhouette data not available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('Mixed-Type Cluster Quality (Gower Silhouette - N/A)')
plt.xticks(results_dataframe['k'].values)

plt.tight_layout()
plt.show()

# --- Final Model Training ---
print(f"\nTraining final model using k={optimal_k}")

# Load full clean dataset (main_data_scaled already contains the scaled numerical and original categorical data)
full_clean_data = main_data_scaled[numerical_features + categorical_features].copy()
for col in categorical_features:
    if col in full_clean_data.columns: # Ensure column exists before converting
        full_clean_data[col] = full_clean_data[col].astype(str)
    else:
        print(f"Warning: Categorical feature '{col}' not found in full_clean_data for final model training.")


# Recompute categorical indices for full_clean_data
recompute_categorical_indices = [full_clean_data.columns.get_loc(col) for col in categorical_features if col in full_clean_data.columns]

# Final model Train
final_kproto = KPrototypes(
    n_clusters=optimal_k,
    init='Cao',
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=n_jobs
)

start_time = time.time()
final_model_clean_clusters = final_kproto.fit_predict(full_clean_data, categorical=recompute_categorical_indices)
training_time = time.time() - start_time

print(f"Final Training Time completed in {training_time:.1f} seconds")

# --- Assign clusters ---
main_data['cluster'] = final_model_clean_clusters
# Using .loc for explicit assignment to avoid SettingWithCopyWarning
outlier_data.loc[:, 'cluster'] = -1  # Assign outliers to special character (-1)

# Combining results
final_dataframe = pd.concat([main_data, outlier_data]).copy() # Use .copy() to avoid future warnings

# --- Cluster Analysis ---
# Reverse scaling for interpretation
# Ensure `mads` and `medians` are properly aligned with `numerical_features`
for i, col in enumerate(numerical_features):
    if col in final_dataframe.columns: # Check if column exists in final_dataframe
        mask = final_dataframe['cluster'] != -1
        # Ensure `mads` and `medians` are accessed correctly (by index for mads, by column for medians)
        # Assuming mads is an array and medians is a Series/dict indexed by feature names
        if i < len(mads): # Check if index is within bounds for mads array
            final_dataframe.loc[mask, col] = final_dataframe.loc[mask, col] * mads[i] + medians[col]
        else:
            print(f"Warning: Index {i} out of bounds for mads during inverse scaling of {col}. Skipping.")
    else:
        print(f"Warning: Numerical feature '{col}' not found in final_dataframe for inverse scaling. Skipping.")


# Create cluster labels function
print("\nAuto Creating Cluster label")

def auto_create_clusters_label(cluster_id, df_source, time_map):
    if cluster_id == -1:
        return 'Outlier'
    
    cluster_data = df_source[df_source['cluster'] == cluster_id].copy()

    if cluster_data.empty:
        return f"Cluster {cluster_id} (Empty)"

    # Intensity
    # Ensure 'global_active_power' exists
    if 'global_active_power' in cluster_data.columns:
        power_median = cluster_data['global_active_power'].median()
        if pd.isna(power_median):
            intensity = "Unknown Intensity"
        elif power_median < 0.5:
            intensity = "Low"
        elif power_median < 1.0:
            intensity = "Medium"
        else:
            intensity = "High"
    else:
        intensity = "No Power Data"

    # Time of day
    time_label = 'Unknown Time'
    if 'time_of_day' in cluster_data.columns:
        # Convert time_of_day to numeric, coercing errors to NaN
        cluster_data['time_of_day_numeric'] = pd.to_numeric(cluster_data['time_of_day'], errors='coerce')
        
        # Calculate mode, handling multiple modes or null
        time_mode = cluster_data['time_of_day_numeric'].mode(dropna=True)

        if not time_mode.empty:
            try:
                time_mode_val = int(time_mode.iloc[0])
                # Reverse time_mapping
                reverse_time_mapping = {v: k for k, v in time_map.items()}
                time_label = reverse_time_mapping.get(time_mode_val, f"Invalid ({time_mode_val})")
            except (ValueError, IndexError) as e:
                print(f"Error processing time_of_day mode for cluster {cluster_id}: {str(e)}")
        else:
            print(f"No valid time_of_day mode found for cluster {cluster_id}.")
    else:
        print(f"Warning: 'time_of_day' column not found in cluster data for cluster {cluster_id}.")


    # Appliance Dominated
    appliance = "Balanced"
    if 'kitchen_ratio' in cluster_data.columns and 'ac_ratio' in cluster_data.columns:
        kitchen_median = cluster_data['kitchen_ratio'].median()
        ac_median = cluster_data['ac_ratio'].median()
        
        if pd.isna(kitchen_median) or pd.isna(ac_median):
            appliance = "Unknown Appliance"
        elif kitchen_median > 0.4 and ac_median > 0.4:
            appliance = "Kitchen+Ac Dominated"
        elif kitchen_median > 0.4:
            appliance = "Kitchen Dominated"
        elif ac_median > 0.4:
            appliance = "Ac Dominated"
    else:
        print(f"Warning: 'kitchen_ratio' or 'ac_ratio' not found in cluster data for cluster {cluster_id}.")


    # Weekend pattern (is_weekend)
    temporal = "Mixed"
    if 'is_weekend' in cluster_data.columns:
        weekend_prop = cluster_data['is_weekend'].mean()
        if pd.isna(weekend_prop):
            temporal = "Unknown Temporal"
        elif weekend_prop > 0.6:
            temporal = "Weekend"
        elif weekend_prop < 0.4:
            temporal = "Weekday"
    else:
        print(f"Warning: 'is_weekend' column not found in cluster data for cluster {cluster_id}.")

    return f"{intensity} {time_label} {appliance} {temporal}"

# Apply auto create clusters label
cluster_labels = {}
# Ensure time_mapping is passed to the function
time_mapping_for_labels = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}

for cluster_id in final_dataframe['cluster'].unique():
    cluster_labels[cluster_id] = auto_create_clusters_label(cluster_id, final_dataframe, time_mapping_for_labels)

final_dataframe['cluster_labels'] = final_dataframe['cluster'].map(cluster_labels)
print(final_dataframe.head())
print("\nCluster Label Summary:")
print(final_dataframe['cluster_labels'].value_counts())

# --- Visualize cluster results ---
print("\nVisualizing Cluster Results")

plt.figure(figsize=(14, 8))

# Cluster distribution
plt.subplot(1, 2, 1)
sns.countplot(x='cluster_labels', data=final_dataframe, order=final_dataframe['cluster_labels'].value_counts().index)
plt.title('Cluster Distribution')
plt.xticks(rotation=90)
plt.xlabel('Cluster Labels')
plt.ylabel('Number of Records')


# Hourly Patterns
plt.subplot(1, 2, 2)
# Ensure 'hour' and 'global_active_power' are available for plotting
if 'hour' in final_dataframe.columns and 'global_active_power' in final_dataframe.columns:
    for label in final_dataframe['cluster_labels'].unique():
        if label == "Outlier":
            continue
        label_data = final_dataframe[final_dataframe['cluster_labels'] == label]
        if not label_data.empty: # Only plot if there's data for the label
            hourly_avg = label_data.groupby('hour')['global_active_power'].median()
            plt.plot(hourly_avg, label=label)

    plt.title('Hourly Consumption Pattern by Cluster')
    plt.xlabel('Hour of Day')
    plt.ylabel('Median Global Active Power')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
    plt.grid(True)
else:
    plt.text(0.5, 0.5, 'Hourly pattern visualization not possible: missing "hour" or "global_active_power"', 
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('Hourly Consumption Pattern (N/A)')

plt.tight_layout()
plt.show() # Call plt.show() after all subplots are created

# --- Outliers Results ---
if len(outlier_data) > 0:
    print('\nOutlier Analysis')
    # Ensure 'hour' and 'global_active_power' exist in outlier_data
    if 'hour' in outlier_data.columns and 'global_active_power' in outlier_data.columns:
        outlier_summary = outlier_data.groupby('hour')['global_active_power'].agg(['count', 'median'])
        print(outlier_summary)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='hour', y='global_active_power', data=outlier_data, alpha=0.7)
        plt.title('Outlier Consumption by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Global Active Power')
        plt.grid(True)
        plt.show()
    else:
        print("Skipping outlier visualization: 'hour' or 'global_active_power' column not found in outlier data.")
else:
    print("\nNo outliers detected.")