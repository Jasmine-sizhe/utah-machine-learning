#Imports
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, hamming_loss, f1_score
import torch


#Derive fl_status column based on fl_echo and FLI(when available): 1 for positive; 0 for negative; -1 for unavailable
def derive_fl_status(row):
    # Derive FL_Check column to infer the status by echo or FLI
    liver_status = row['脂肪肝 fatty Liver (0:正常  1:mild 2:moderate 3:severe)']
    fli_value = row['FLI']

    if pd.isna(liver_status) and pd.isna(fli_value):
        return -1
    elif pd.notna(liver_status) and liver_status != 0:
        return 1
    elif pd.notna(fli_value) and fli_value >= 60:
        return 1
    else:
        return 0

# Derive CKD status
def derive_CKD(df):
    # Initialize CKD field as -1
    df['CKD'] = -1

    # Condition 1: egfrn >= 60 and Alb_Cre_ratio < 3
    df.loc[(df['Estimated_GFR_x'] >= 60) & (df['Alb_Cre_ratio'] < 3), 'CKD'] = 1

    # Condition 2: egfrn >= 60 and 3 <= Alb_Cre_ratio <= 30 or 45 <= egfrn < 60 and Alb_Cre_ratio < 3
    df.loc[((df['Estimated_GFR_x'] >= 60) & (df['Alb_Cre_ratio'].between(3, 30))) |
           ((df['Estimated_GFR_x'].between(45, 60)) & (df['Alb_Cre_ratio'] < 3)), 'CKD'] = 2

    # Condition 3: egfrn >= 60 and Alb_Cre_ratio > 30 or egfrn < 60 and Alb_Cre_ratio >= 0
    df.loc[((df['Estimated_GFR_x'] >= 60) & (df['Alb_Cre_ratio'] > 30)) |
           ((df['Estimated_GFR_x'] < 60) & (df['Alb_Cre_ratio'] >= 0)), 'CKD'] = 3

    # Set CKD as 0 for cases where egfrn and Alb_Cre_ratio are not empty and CKD is still -1
    df.loc[(df['Estimated_GFR_x'].notnull()) & (df['Alb_Cre_ratio'].notnull()) & (df['CKD'] == -1), 'CKD'] = 0

    return df

def derive_MAFLD(df):
    df['MAFLD'] = 0  # Initialize MAFLD field as 0

    # Condition 1: fl_status = -1
    df.loc[df['fl_status'] == -1, 'MAFLD'] = -1

    # Condition 2: fl_check = 0
    df.loc[df['fl_status'] == 0, 'MAFLD'] = 0

    # Condition 3: fl_check = 1
    # Subcondition 1: BMI >= 23
    df.loc[(df['fl_status'] == 1) & (df['BMI'] >= 23), 'MAFLD'] = 1

    # Subcondition 2: BMI < 23 and mst >= 2
    df.loc[(df['fl_status'] == 1) & (df['BMI'] < 23) & (df['mst_total'] >= 2), 'MAFLD'] = 1

    # Subcondition 3: DM_determine = 1
    df.loc[(df['fl_status'] == 1) & (df['DM_determine'] == 1), 'MAFLD'] = 1

    return df

# We are now deriving target variables as MAFLD_0, MAFLD_Obesity, MAFLD_Diebetes, MAFLD_MD
def derive_MAFLD_with_multi_label(df):
    # Condition for non-case: derive MAFLD_0, if MAFLD is 0, then MAFLD_0 is 1, otherwise 0
    df['MAFLD_0'] = 0
    df.loc[df['MAFLD'] == 0, 'MAFLD_0'] = 1

    # Condition 1: MAFLD_Obesity, if MAFLD is 1 and BMI >= 23
    df['MAFLD_Obesity'] = 0
    df.loc[(df['MAFLD'] == 1) & (df['BMI'] >= 23), 'MAFLD_Obesity'] = 1

    # Condition 2: MAFLD_MD, if MAFLD is 1 and BMI < 23 and mst >= 2
    df['MAFLD_MD'] = 0
    df.loc[(df['MAFLD'] == 1) & (df['BMI'] < 23) & (df['mst_total'] >= 2), 'MAFLD_MD'] = 1

    # Condition 3: MAFLD_Diabetes, if MAFLD is 1 and DM_determine = 1
    df['MAFLD_Diabetes'] = 0
    df.loc[(df['MAFLD'] == 1) & (df['DM_determine'] == 1), 'MAFLD_Diabetes'] = 1

    # Additional Condition: if MAFLD is -1, set all labels to -1
    df.loc[df['MAFLD'] == -1, ['MAFLD_0', 'MAFLD_Obesity', 'MAFLD_MD', 'MAFLD_Diabetes']] = -1

    return df

# Derive patient_fl_validity according to FL_group_list conditions
def assign_patient_fl_validity(df):
    df['patient_fl_validity'] = -1  # 初始化所有记录为第三组 (-1)

    def get_patient_valid(group_list):
        if isinstance(group_list, list):
            if -1 in group_list and len(group_list) == 1:
                return "unavailable"  # 第三组 (-1)
            elif -1 in group_list:
                return "partial"  # 第二组 (0)
            else:
                return "completed"  # 第一组 (1)
        else:
            return "others"  # 第三组 (-1)

    df['patient_fl_validity'] = df['fl_group_list'].apply(get_patient_valid)

    return df

# This function will extract the numeric values
import re
def extract_numeric_value(value):
    pattern = r'\d+(\.\d+)?'  # 正则表达式模式，匹配一个或多个数字（包括小数点）
    match = re.search(pattern, str(value))
    if match:
        return float(match.group())
    else:
        return None

# Sliding window function
def sliding_window_data(df, input_window_size, target_window_size):
    transformed_data = []
    group_counter = {}

    df_sorted = df.sort_values(['CMRC_id', 'year_come'])

    for patient_id, group in df_sorted.groupby('CMRC_id'):
        if len(group) < input_window_size + target_window_size:
            continue


        group_counter.setdefault(patient_id, 0)
        group_counter[patient_id] += 1
        group_alias = f'{patient_id}_group{group_counter[patient_id]}'

        for i in range(len(group) - input_window_size - target_window_size + 1):
            input_data = group[i:i+input_window_size]
            target_data = group[i+input_window_size:i+input_window_size+target_window_size]

            # Flatten input_data and repeat target_data
            input_features_t1 = input_data.iloc[0, :].values.flatten()
            input_features_t2 = input_data.iloc[1, :].values.flatten()
            t3_MAFLD = target_data['MAFLD'].values

            new_row = [group_alias] + list(input_features_t1) + list(input_features_t2) + list(t3_MAFLD)

            transformed_data.append(new_row)

    columns_list = ['CMRC_id'] + [f't1_{col}' for col in input_data.columns] + [f't2_{col}' for col in input_data.columns] + [f't3_MAFLD']
    transformed_df = pd.DataFrame(transformed_data, columns=columns_list)
    return transformed_df

# This function is for adding prefix for cols, the cols should be a list of column names that needs to add prefix(such as "t1_" in this project)
def add_prefix(cols, prefixes):
# Note the prefixes should be a LIST, eg. prefixes = ["t1_", "t2_"]
    renamed_columns = []
    for prefix in prefixes:
        renamed_columns.extend([prefix + column for column in cols])
    return renamed_columns


def remove_columns_with_high_missing_values(df, threshold):
    """
    Remove columns from a DataFrame that have missing values exceeding the specified threshold.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - threshold (float): The threshold for missing values. Columns with missing values exceeding this threshold will be removed.

    Returns:
    - cleaned_df (DataFrame): The cleaned DataFrame with columns removed.
    """
    total_missing = df.isnull().sum()  # 计算每列的缺失值数量
    total_rows = df.shape[0]  # 数据集的总行数
    columns_to_remove = total_missing[total_missing / total_rows > threshold].index  # 找到超过阈值的列名
    cleaned_df = df.drop(columns=columns_to_remove)  # 删除指定列
    print("columns to remove with high missing values: ", columns_to_remove)

    return cleaned_df

def remove_columns_with_high_unique_values(df, threshold):
    """
    Remove columns from a DataFramse that have unique values exceeding the specified threshold.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - threshold (int): The threshold for unique values. Columns with unique values exceeding this threshold will be removed.

    Returns:
    - cleaned_df (DataFrame): The cleaned DataFrame with columns removed.
    """
    categorical_columns = df.select_dtypes(include='object').columns  # 获取分类字段列
    columns_to_remove = []
    for column in categorical_columns:
        unique_values = df[column].nunique()  # 计算唯一值数量
        if unique_values > threshold:
            columns_to_remove.append(column)
    cleaned_df = df.drop(columns=columns_to_remove)  # 删除指定列

    return cleaned_df

def generate_column_names(prefix, column_names):
    new_column_names = []
    for p in prefix:
        new_column_names.extend([p + col for col in column_names])
    return new_column_names


def analyze_numeric_column(df, column_name, output_file):
    non_numeric_values = df[column_name][~df[column_name].apply(lambda x: isinstance(x, (int, float)))]
    unique_non_numeric_values = non_numeric_values.unique()
    non_numeric_ratio = len(non_numeric_values) / len(df[column_name])
    null_ratio = df[column_name].isnull().mean()
    
    output = f"Column: {column_name}\n"
    output += "Non-numeric Values:\n" + str(unique_non_numeric_values) + "\n"
    output += "Non-numeric Ratio: " + str(non_numeric_ratio) + "\n"
    output += "Null Ratio: " + str(null_ratio) + "\n"
    output += "\n"
    
    with open(output_file, "a") as file:
        file.write(output)

def save_categorical_unique_values(df, categorical_features, output_file):
    with open(output_file, "w") as file:
        for feature in categorical_features:
            unique_values = df[feature].unique()
            output = f"Column: {feature}\n"
            output += "Unique Values:\n" + str(unique_values) + "\n\n"
            file.write(output)
            
## Additional Adding NFS and Fibrosis-4 score
def calculate_FIB_4_Score(row):
    return (row['age'] * row['AST_GOT']) / (row['Platelets'] * np.sqrt(row['ALT_GPT']))

def calculate_NFS(row):
    return -1.675 + (0.037 * row['age']) + (0.094 * row['BMI']) + (1.13 * row['DM_determine']) + (0.99 * np.log10(row['AST_GOT'] / row['ALT_GPT'])) + (0.013 * row['Platelets']) - (0.66 * row['Albumin'])

def select_columns(df, prefix, additional_column=None):
    selected_columns = [col for col in df.columns if col.startswith(prefix)]
    if additional_column is not None:
        selected_columns.append(additional_column)
    return df[selected_columns]

# Sliding window function for multi_label
def sliding_window_multi_label_data(df, input_window_size, target_window_size):
    transformed_data = []
    group_counter = {}

    df_sorted = df.sort_values(['CMRC_id', 'year_come'])

    for patient_id, group in df_sorted.groupby('CMRC_id'):
        if len(group) < input_window_size + target_window_size:
            continue

        group_counter.setdefault(patient_id, 0)
        group_counter[patient_id] += 1
        group_alias = f'{patient_id}_group{group_counter[patient_id]}'

        for i in range(len(group) - input_window_size - target_window_size + 1):
            input_data = group[i:i+input_window_size]
            target_data = group[i+input_window_size:i+input_window_size+target_window_size]

            # Flatten input_data and repeat target_data
            input_features_t1 = input_data.iloc[0, :].values.flatten()
            target_columns = [f't2_{col}' for col in target_data.columns]

            new_row = [group_alias] + list(input_features_t1) + list(target_data.values[0])
            
            transformed_data.append(new_row)

    columns_list = ['CMRC_id'] + [f't1_{col}' for col in input_data.columns] + [f't2_{col}' for col in target_data.columns]
    
    transformed_df = pd.DataFrame(transformed_data, columns=columns_list)
    return transformed_df

# feature processing for scaling and missing imputation
def preprocess_features_and_target(df):
    # Drop the 't1_MAFLD_0' column if it exists
    if 't1_MAFLD_0' in df.columns:
        df = df.drop('t1_MAFLD_0', axis=1)

    features = df.columns.drop(['t2_MAFLD_0', 't2_MAFLD_Obesity', 't2_MAFLD_Diabetes', 't2_MAFLD_MD'])
    categorical_features = ['t1_sex', 't1_w', 't1_smoke', 't1_smoke_q', 't1_coffee', 't1_betel', 't1_DM_determine', 't1_CKD']
    numeric_features = df.columns.drop(categorical_features).drop(['t2_MAFLD_0', 't2_MAFLD_Obesity', 't2_MAFLD_Diabetes', 't2_MAFLD_MD'])

    # Split the DataFrame into categorical and numeric DataFrames
    X_categorical = df[categorical_features]
    X_numeric = df[numeric_features]

    # Define the target variable
    y = df[['t2_MAFLD_0', 't2_MAFLD_Obesity', 't2_MAFLD_Diabetes', 't2_MAFLD_MD']]

    # Scale numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # Handle missing values in numeric features
    imputer = SimpleImputer(strategy='median')
    X_numeric_scaled_imputed = imputer.fit_transform(X_numeric_scaled)

    # Encode categorical features into dummy variables
    X_categorical_str = X_categorical.astype(str)
    X_categorical_encoded = pd.get_dummies(X_categorical_str, drop_first=True)

    # Concatenate scaled numeric features and encoded categorical features
    # concat
    X_numeric_scaled_imputed = pd.DataFrame(X_numeric_scaled_imputed, columns=X_numeric.columns)
    X_numeric_scaled_imputed.reset_index(drop=True, inplace=True)
    X_categorical_encoded.reset_index(drop=True, inplace=True)
    X_combined = pd.concat([X_numeric_scaled_imputed, X_categorical_encoded], axis=1)

    return X_combined, y

def calculate_performance_AEclassifier(val_loader, model):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            # print(outputs)
            # break  #check for one batch
            predicted = outputs[1].data
            all_labels.append(labels.numpy())
            all_predictions.append(predicted.numpy())

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions > 0.5)
    f1_micro = f1_score(all_labels, all_predictions > 0.5, average='micro')
    f1_macro = f1_score(all_labels, all_predictions > 0.5, average='macro')
    hl = hamming_loss(all_labels, all_predictions > 0.5)

    auc_micro = roc_auc_score(all_labels, all_predictions, average='micro')
    auc_macro = roc_auc_score(all_labels, all_predictions, average='macro')

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'F1 Score (Micro): {f1_micro:.4f}')
    print(f'F1 Score (Macro): {f1_macro:.4f}')
    print(f'Hamming Loss: {hl:.4f}')
    print(f'AUC (Micro): {auc_micro:.4f}')
    print(f'AUC (Macro): {auc_macro:.4f}')

    auc_scores = roc_auc_score(all_labels, all_predictions, average=None)
    targets = ['t2_MAFLD_0', 't2_MAFLD_Obesity', 't2_MAFLD_Diabetes', 't2_MAFLD_MD']

    for i, target in enumerate(targets):
        auc = auc_scores[i]
        print(f'AUC for {target}: {auc:.4f}')

    return accuracy, f1_micro, f1_macro, hl, auc_micro, auc_macro, auc_scores