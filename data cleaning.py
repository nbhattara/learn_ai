import pandas as pd

#  Load your raw data
# Choose the method that matches your file type
df = pd.read_csv('messy_sales_data.csv')           
# df = pd.read_excel('raw_data.xlsx', sheet_name='Sheet1')
# df = pd.read_parquet('large_dataset.parquet')
#df = pd.read_sql("SELECT * FROM sales", con=your_database_connection)

# (Optional but strongly recommended) Quick look at the mess
print("Initial data overview:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

#  Run the cleaning function
# Customize the parameters according to your dataset
cleaned_df = clean_dataset(
    df,
    date_columns=['order_date', 'signup_date', 'last_activity'],      # ← date columns here
    numeric_columns=['price', 'quantity', 'discount', 'revenue'],     # ← explicit numeric columns 
    drop_duplicates_subset=['order_id', 'customer_id', 'product_id'], # ← most important for business data
    inplace=False  # keep original safe
)

#  Verify the result
print("\nAfter cleaning:")
print(cleaned_df.info())
print("\nRemaining missing values:")
print(cleaned_df.isnull().sum())

# Savecleaned version
cleaned_df.to_csv('cleaned_sales_2026-01-04.csv', index=False)
cleaned_df.to_parquet('cleaned_sales.parquet')   # faster & smaller - recommended for larger files
cleaned_df.to_excel('cleaned_report.xlsx', index=False)