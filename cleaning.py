import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sqlalchemy import create_engine
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import StandardScaler

def extract_data(filename:str, output_path:str):
    '''
    Extract sales data from csv file and save it to a new csv file
    @param file_path: str, path to the csv file
    @param output_path: str, path to save the new csv file
    '''

    df_fintech = pd.read_csv(filename)
    df_fintech1 = renamecol_index(df_fintech)
    df_fintech2 = date_length_grade_term_StandardizedTypeColum_checkDuplicates_checkmissing(df_fintech1, 'issue_date', 'emp_length', 'grade', 'term', 'type')
    df_fintech3 = Annual_income_EDA(df_fintech2, 'annual_inc', 'home_ownership')
    df_fintech4 = handle_missing(df_fintech3, 'emp_title','int_rate','description','annual_inc_joint')
    df_fintech5 =grade_analysis(df_fintech4,'annual_income_grouped','grade_grouped')
    df_fintech6=salary_can_cover(df_fintech5)
    df_fintech6.to_parquet(output_path)

def extract_states(filename:str, output_path:str):
    '''
    Transform sales data by imputing missing values and encoding categorical columns
    @param filename: str, path to the csv file
    @param output_path: str, path to save the new csv file
    '''
    df_fintech = pd.read_csv(filename)
    df_fintech.rename(columns={'code':'state'},inplace=True)
    df_fintech.to_parquet(output_path)

def combine_sources(filename:str, filename1:str, output_path:str):
    df_fintech=pd.read_parquet(filename)
    df_states=pd.read_parquet(filename1)
    df_fintech_combined=pd.merge(df_fintech, df_states, on='state', how='inner', left_on=None, right_on=None)
    df_fintech_combined.to_parquet(output_path)

def encoding(filename:str,output_path:str):
    df_fintech=pd.read_parquet(filename)
    df_fintech=label_encoding_function(df_fintech,'home_ownership','verification_status')
    df_fintech =one_hot_encode(df_fintech, 'loan_status')
    df_fintech.to_parquet(output_path)

def load_to_db(filename:str, table_name:str, postgres_opt:dict):
    '''
    Load the transformed data to the database
    @param filename: str, path to the csv file
    @param table_name: str, name of the table to create
    @param postgres_opt: dict, dictionary containing postgres connection options (user, password, host,port, db)
    '''
    user, password, host, port, db = postgres_opt.values()
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}')
    df = pd.read_parquet(filename)
    # Set the index to invoice_id
    df.set_index(['customer_id', 'loan_id'], inplace=True)
    df.to_sql(table_name, con=engine, if_exists='replace', index=True, index_label=['customer_id','loan_id'])

# ---- Helper Functions ----

def renamecol_index(df):
#     make all cols lower case
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # df.set_index(['customer_id','loan_id'], inplace=True)

    return df

def date_length_grade_term_StandardizedTypeColum_checkDuplicates_checkmissing(df, col1, col2, col3, col4, col5):
    """
    Standardizes and transforms date, employment length, grade, and term columns.
    Extracts the issue year from the date column and saves it to a new column.
    Checks for duplicates and missing values, and saves lookup tables for transformations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The column name for the issue date.
        col2 (str): The column name for employment length.
        col3 (str): The column name for grade.
        col4 (str): The column name for term.
        col5 (str): The column name for type.

    Returns:
        None
    """
    # Standardize the 'type' column
    look_Df_type = df[[col5]].copy()
    look_Df_type.rename(columns={col5: 'original_type_column'}, inplace=True)
    type_mapping = {
        'Individual': 'Individual',
        'INDIVIDUAL': 'Individual',
        'Joint App': 'Joint App',
        'JOINT': 'Joint App',
        'DIRECT_PAY': 'Direct Pay',
        'DIRECT PAY': 'Direct Pay'
    }
    df[col5] = df[col5].map(type_mapping).fillna(df[col5])
    look_Df_type['Standarized_type_column'] = df[col5]
    look_Df_type.to_csv('Standarized_type_column.csv', index=True)
    print("Standarized_type_column.csv saved.")

    # Ensure the issue date column is in datetime format
    df[col1] = pd.to_datetime(df[col1])

    # Extract the issue year and save it to a new column
    df['issue_year'] = df[col1].dt.year

    # Create a new column for the issue month
    df['issue_month'] = df[col1].dt.month

    # Sort by the issue date
    df.sort_values(by=[col1], inplace=True)

    # Create a lookup DataFrame for the issue date
    lookup_df = df[[col1, 'issue_month', 'issue_year']].copy()
    lookup_df.rename(columns={col1: 'transformed_issue_date'}, inplace=True)
    lookup_df['original_issue_date'] = df[col1].astype(str)
    lookup_df.to_csv('date_lookup_table.csv', index=False)
    print("Lookup table saved to date_lookup_table.csv")

    # Transform the employment length column
    lookup_df1 = df[[col2]].copy()
    lookup_df1.rename(columns={col2: 'original_emp_length'}, inplace=True)
    replacements = {'< 1 year': '1', '10+ years': '10'}
    df[col2] = df[col2].replace(replacements, regex=False)
    df[col2] = pd.to_numeric(
        df[col2].str.extract(r'(\d+\.?\d*)')[0],  # Extract numerical part
        errors='coerce'  # Convert invalid parsing to NaN
    )
    lookup_df1['emp_length_numeric'] = df[col2]
    lookup_df1.to_csv('emp_lengthTransformation_lookup.csv', index=False)
    print("Lookup table saved to emp_lengthTransformation_lookup.csv")

    # Transform the term column
    lookup_df3 = df[[col4]].copy()
    lookup_df3.rename(columns={col4: 'original_term'}, inplace=True)
    replacements1 = {'36 months': '36', '60 months': '60'}
    df[col4] = df[col4].replace(replacements1, regex=False)
    df[col4] = pd.to_numeric(
        df[col4].str.extract(r'(\d+\.?\d*)')[0],  # Extract numerical part
        errors='coerce'  # Convert invalid parsing to NaN
    )
    lookup_df3['Term_numeric'] = df[col4]
    lookup_df3.to_csv('term_numeric_lookup.csv', index=False)
    print("Lookup table saved to term_numeric_lookup.csv")

    # Group the grade column
    lookup_df2 = df[[col3]].copy()
    lookup_df2.rename(columns={col3: 'original_grade'}, inplace=True)
    Y = [
        (df[col3] >= 1) & (df[col3] <= 5),
        (df[col3] >= 6) & (df[col3] <= 10),
        (df[col3] >= 11) & (df[col3] <= 15),
        (df[col3] >= 16) & (df[col3] <= 20),
        (df[col3] >= 21) & (df[col3] <= 25),
        (df[col3] >= 26) & (df[col3] <= 30),
        (df[col3] >= 31) & (df[col3] <= 35)
    ]
    grouping = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df['grade_grouped'] = np.select(Y, grouping, default=None)
    lookup_df2['grades_grouped'] = df['grade_grouped']
    lookup_df2.to_csv('grade_lookup.csv', index=False)
    print("Lookup table saved to grade_lookup.csv")

    # Check for duplicates
    duplicates = df[df.duplicated(subset=None)]
    if not duplicates.empty:
        duplicates.to_csv('duplicate_rows_lookup.csv', index=False)
        print("Duplicate rows saved to duplicate_rows_lookup.csv")
    else:
        print("No duplicate rows found.")

    # Drop duplicates
    df.drop_duplicates(subset=None, inplace=True)

    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': missing_counts,
        'Percentage (%)': missing_percentages
    })
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]

    if not missing_summary.empty:
        missing_summary.to_csv('missing_values_lookup.csv', index=False)
        print("Missing values lookup table saved to missing_values_lookup.csv")
    else:
        print("No missing values found.")

    print(missing_summary)
    return df

    

def Annual_income_EDA(df, col, categorical_col):
    lookup_df = df[[col]].copy()
    lookup_df.rename(columns={col: 'original_annual_income'}, inplace=True)

    # Calculate basic statistics
    mean_annual_inc = df[col].mean()
    median_annual_inc = df[col].median()
    mode_annual_inc = df[col].mode()[0]
    max_annual_inc = df[col].max()
    min_annual_inc = df[col].min()
    max_annual_inc_3 = df[col].nlargest(10).tolist()
    min_annual_inc_3 = df[col].nsmallest(10).tolist()

    # Print statistics
    print("Annual income median:", median_annual_inc.round())
    print("Annual income mean:", mean_annual_inc.round())
    print("Annual income mode:", mode_annual_inc.round())
    print("Annual income Max:", max_annual_inc)
    print("Annual income Min:", min_annual_inc)
    print("Annual income Max 5 values:", max_annual_inc_3)
    print("Annual income Min 5 values:", min_annual_inc_3)

    # Create income groups
    Y = [
        (df[col] >= 0) & (df[col] < 35000),
        (df[col] >= 35000) & (df[col] < 60000),
        (df[col] >= 60000) & (df[col] < 100000),
        (df[col] >= 100000) & (df[col] < 200000),
        (df[col] >= 200000) & (df[col] < 500000),
        (df[col] >= 500000) & (df[col] <= 3000000)
    ]
    grouping = ['Lower_Class', 'Working_Class', 'Lower_Middle_Class', 
                'Upper_Middle_Class', 'Upper_Class', 'Top_1%']
    df['annual_income_grouped'] = np.select(Y, grouping, default=None)
    df['annual_income_grouped'] = pd.Categorical(df['annual_income_grouped'], categories=grouping, ordered=True)
    
    # Save lookup table
    lookup_df['annual_income_grouped'] = df['annual_income_grouped']
    lookup_df.to_csv('annual_income_lookupTable.csv', index=False)

    # Calculate percentage of each income group
    income_group_counts = df['annual_income_grouped'].value_counts(normalize=False).sort_index()
    income_group_percentages = df['annual_income_grouped'].value_counts(normalize=True).sort_index() * 100

    # Print percentages
    print("\nPercentage of each income group from the total dataset:")
    for group, percentage in zip(grouping, income_group_percentages):
        print(f"{group}: {percentage:.2f}%")
    
    # Optional: Save the percentages to a CSV file
    percentage_df = pd.DataFrame({
        'Income Group': grouping,
        'Count': income_group_counts.values,
        'Percentage': income_group_percentages.values
    })
    percentage_df.to_csv('income_group_percentages.csv', index=False)

    print(df['home_ownership'].value_counts())
    
    print("\nIncome group percentages saved to 'income_group_percentages.csv'.")

    # Create the histogram with larger size
    # plt.figure(figsize=(15, 8))
    # ax = sns.histplot(data=df, x='annual_income_grouped', palette='Set2', discrete=True)

    # Add count labels above each bar
    # for bar, count in zip(ax.patches, income_group_counts):
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,  # x position
    #         bar.get_height() + 1,  # y position (slightly above the bar)
    #         f"{count}",  # Label
    #         ha='center', fontsize=10, color='black'
    #     )

    # # Set plot titles and labels
    # plt.title('Annual Income Distribution by Group', fontsize=16)
    # plt.xlabel('Income Group', fontsize=14)
    # plt.ylabel('Count', fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.show()

    # Analyze and print the categorical column if provided
    if categorical_col in df.columns:
        Annual_income_ownership1 = (
            df.groupby('annual_income_grouped')[categorical_col]
            .value_counts(normalize=False)  # Get counts
            .rename('count')
            .reset_index()
        )

        # Print the counts for each group
        print("\nCounts of each homeowner group in each annual income class:")
        for income_group, group_data in Annual_income_ownership1.groupby('annual_income_grouped'):
            print(f"\nIncome Group: {income_group}")
            for index, row in group_data.iterrows():
                print(f"  {row[categorical_col]}: {row['count']}")

        # Add percentage calculations
        Annual_income_ownership1['percentage'] = (
            Annual_income_ownership1.groupby('annual_income_grouped')['count']
            .transform(lambda x: x / x.sum() * 100)
        )

        # Save the categorical analysis
        Annual_income_ownership1.to_csv(f'{categorical_col}_income_group_analysis.csv', index=False)

        # Plot the categorical column percentages only
        # plt.figure(figsize=(20, 10))
        # ax = sns.barplot(
        #     data=Annual_income_ownership1, 
        #     x='annual_income_grouped', 
        #     y='percentage', 
        #     hue=categorical_col,
        #     palette='Set3'
        # )

        # Add percentage labels above bars
        # for bar in ax.containers:
        #     ax.bar_label(bar, fmt='%.1f%%', fontsize=9, label_type='edge')

        # plt.title(f'Percentage of {categorical_col} Categories by Annual Income Group', fontsize=16)
        # plt.ylabel('Percentage', fontsize=12)
        # plt.xlabel('Annual Income Group', fontsize=12)
        # plt.legend(title=categorical_col, fontsize=10)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.show()
    else:
        print(f"Categorical column '{categorical_col}' not found in the dataset. Skipping related analysis.")
      
    return df
def one_hot_encode(df_1, column, prefix=None):
 
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    
    # Fit and transform the specified column
    encoded_array = encoder.fit_transform(df_1[[column]])
    
    # Create column names
    if prefix is None:
        prefix = column
    encoded_columns = [f"{prefix}_{category}" for category in encoder.categories_[0]]
    
    # Convert the encoded array to a DataFrame
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df_1.index)
    
    # Concatenate the encoded DataFrame with the original DataFrame (drop original column)
    df_1 = pd.concat([df_1, encoded_df], axis=1)
    
    return df_1

def label_encoding_function(df,col,col1,col2=''): 
 
#  lookup_df = df[[col]].copy()
#  lookup_df.rename(columns={col: 'original_home_ownership'}, inplace=True)
#  lookup_df1 =df[[col1]].copy()
#  lookup_df1.rename(columns={col1: 'original_Verification_status'}, inplace=True)
# Custom class order
 desired_classes = ['MORTGAGE', 'RENT', 'OWN', 'ANY', 'OTHER']
 verification_mapping = {
    'Not Verified': 0,
    'Verified': 1,
    'Source Verified': 2
}
# Initialize LabelEncoder
 label_encoder = LabelEncoder()
 label_encoder.classes_ = np.array(desired_classes)
 df['Verification_Status_Encoded'] = df[col1].map(verification_mapping)

# Encode column
 df['Encoded_home_ownership'] = df[col].apply(
    lambda x: label_encoder.transform([x])[0] if x in desired_classes else np.nan
)
 
 
 
#  lookup_df['Encoded_home_ownership'] = df['Encoded_home_ownership']
#  lookup_df.to_csv('Encoded_home_ownership_lookupTable.csv', index=False)
#  lookup_df1['Verification_Status_Encoded'] = df['Verification_Status_Encoded']
#  lookup_df1.to_csv('Verification_Status_Encoded_lookupTable.csv', index=False)
 return df


def handle_missing(df,col1,col2,col3,col4,emp_length_col='emp_length',grade='grade',purpose='purpose',annual_group='annual_income_grouped'):
    
    # Group by emp_title and calculate the mean annual income
    mean_salaries_by_title = df.groupby('emp_title')['annual_inc'].mean()
    title_counts = df['emp_title'].value_counts()
    # plt.figure(figsize=(10, 8))
    # sns.histplot(df[col2],kde=True)
    # plt.title("distribution before imputing")
    # # Create a lookup table for original and imputed titles
    # lookup_df=df[[col2]].copy()
    # lookup_df.rename(columns={col2:'original_int_rate'},inplace=True)

    # lookup_df3 = df[[col1, 'annual_inc']].copy()
    # lookup_df3.rename(columns={col1: 'original_emp_title'}, inplace=True)

    # look_df4=df[[col3]].copy()
    # look_df4.rename(columns={col3: 'original_description'},inplace = True)
    # # Impute missing values in the description column
    df[col3] = df[col3].fillna(df[purpose])
    # look_df4['description_imputed']=df[col3]
    # look_df4.to_csv('lookup_table_description.csv', index=False)

    annual_joint_df=df[[col4]].copy()
    annual_joint_df.rename({col4:'original_annual_joint'},inplace=True)
    df[col4].fillna(0, inplace=True)
    annual_joint_df['imputed_annual_income_joint']=df[col4]
    annual_joint_df.to_csv('jointincome_looktable.csv', index=False)
    # Define the function to impute missing emp_title
    def impute_emp_title(row):
        if pd.isna(row[col1]):  # Check if emp_title is missing
            if pd.isna(row[emp_length_col]):
             return "Unemployed"
            # Calculate the absolute difference from mean salaries
            salary_diff = mean_salaries_by_title.sub(row['annual_inc']).abs()
            
            # Get the closest titles by salary
            closest_titles = salary_diff[salary_diff == salary_diff.min()].index
            
            # From these, choose the title with the highest frequency
            if len(closest_titles) > 1:
                # Filter title_counts to only consider closest_titles
                filtered_title_counts = title_counts[title_counts.index.isin(closest_titles)]
                return filtered_title_counts.idxmax()
            elif len(closest_titles) == 1:
                return closest_titles[0]

            # Fallback if no valid title is found
            return title_counts.idxmax()
        
        return row[col1]  # If not missing, return the original emp_title

    # Apply the imputation function row-wise
    df[col1] = df.apply(impute_emp_title, axis=1)
    df[col2] = df.groupby(grade)[col2].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    # Impute 'emp_length' within each income group
    emp_length_df=df[[emp_length_col]].copy()
    emp_length_df.rename({emp_length_col:'original_emp_length'},inplace=True)
    df[emp_length_col] = df.groupby(annual_group)[emp_length_col].transform(lambda x: x.fillna(x.mode()[0]))
    emp_length_df['imputed_emp_length']=df[emp_length_col]
    emp_length_df.to_csv('emplength_looktable.csv',index=False)
    # Update the lookup table
    # lookup_df['int_rate_imputed']=df[col2]
    # lookup_df.to_csv('int_rate_imputed_lookup.csv', index=False)
    # lookup_df3['emp_title_impute'] = df[col1]
    # # Save the lookup table to a CSV file
    # lookup_df3.to_csv('emp_title_lookup.csv', index=False)
    # # plt.figure(figsize=(10, 8))
    # sns.histplot(df[col2],kde=True)
    # plt.title("distribution after imputing")
    
    print("int_rate_imputed_lookup.csv saved ")
    print("emp_title_lookup.csv saved ")
    print("emplength_looktable.csv saved ")
    print("jointincome_looktable.csv saved ")
    print("lookup_table_description.csv saved ")
    # Return the updated DataFrame
    return df
#This approach ensures that imputation:
#Aligns with the closest mean salary.
#Prefers the most frequent title within the group, improving data consistency.


def grade_analysis(df, income_group_col, grade_col):
    """
    Calculates the percentage of each grade within each annual income group and plots the results.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns.
    income_group_col (str): The name of the column representing annual income groups.
    grade_col (str): The name of the column representing grades.

    Returns:
    pd.DataFrame: A DataFrame with annual income groups, grades, and their percentages.
    """
    # Check if the specified columns exist in the DataFrame
    if income_group_col not in df.columns or grade_col not in df.columns:
        raise ValueError(f"Columns '{income_group_col}' or '{grade_col}' not found in the DataFrame.")
    
    # Calculate count of grades within each income group
    grade_counts = (
        df.groupby([income_group_col, grade_col])
        .size()
        .rename("count")
        .reset_index()
    )

    # Calculate total count of all grades per income group
    total_counts = (
        df.groupby(income_group_col)
        .size()
        .rename("total_count")
        .reset_index()
    )

    # Merge grade counts with total counts
    merged = pd.merge(grade_counts, total_counts, on=income_group_col)

    # Calculate percentage
    merged["percentage"] = (merged["count"] / merged["total_count"]) * 100

    # Sort for better readability
    merged = merged.sort_values(by=[income_group_col, grade_col]).reset_index(drop=True)
    merged.to_csv('grade_analysis.csv', index=False)
    print(f"Grade analysis saved to {'grade_analysis.csv'}")
    # Plot the results
    # plt.figure(figsize=(30, 12))
    # barplot = sns.barplot(
    #     data=merged,
    #     x=income_group_col,
    #     y="percentage",
    #     hue=grade_col,
    #     palette="Set2"
    # )

    # Add percentages above bars
    # for container in barplot.containers:
    #     barplot.bar_label(
    #         container,
    #         fmt='%.1f%%',
    #         label_type='edge',
    #         fontsize=10,
    #         padding=3
    #     )

    # Customize plot
    # plt.title("Percentage of Each Grade by Annual Income Group", fontsize=16)
    # plt.xlabel("Annual Income Group", fontsize=18)
    # plt.ylabel("Percentage", fontsize=18)
    # plt.xticks(rotation=0, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(title="Grade", fontsize=12, title_fontsize=14)
    # plt.tight_layout()
    # plt.show()



    return df

def salary_can_cover(df, P='funded_amount', term_column='term'):
    """
    Calculates the monthly installment for each loan and determines if the borrower's salary can cover it.

    Args:
        df: The input DataFrame containing loan details.
        P: The loan principal/amount (funded amount). Default is 'funded_amount'.
        term_column: The column name containing the term of the loan in months. Default is 'term'.

    Returns:
        A modified DataFrame with two new columns:
        - 'Installment_per_month': Monthly installment for the loan.
        - 'salary_can_cover': Boolean column indicating if annual income can cover yearly installments.
    """
    # Calculate monthly interest rate
    df['monthly_interest_rate'] = df['int_rate'] / 12  

    # Calculate monthly installment
    df['Installment_per_month'] = df.apply(
        lambda row: (
            (row[P] * row['monthly_interest_rate'] * (1 + row['monthly_interest_rate']) ** row[term_column])
            / ((1 + row['monthly_interest_rate']) ** row[term_column] - 1)
        ) if row['monthly_interest_rate'] > 0 else row[P] / row[term_column],
        axis=1
    )

    # Determine if the borrower's salary can cover yearly installments
    df['salary_can_cover'] = (df['annual_inc'] >= df['Installment_per_month'] * 12)

    # Create a lookup DataFrame with the required columns
    lookup_df = df[['Installment_per_month', 'salary_can_cover']]

    # Save the lookup DataFrame to a CSV file
    lookup_df.to_csv('Installmentpermonth_Salarycancover.csv', index=False)
    print(df.groupby('salary_can_cover')['annual_income_grouped'].value_counts())

    return df

