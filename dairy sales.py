import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Read the dataset
file_path = r'C:\Users\User\Desktop\constantinos\sports data analytics\2.Sports Analytics School - Programming Full Course Training 2023\9.Practical Examples on data analysis\datasets\dairy sales\dairy.csv'
df = pd.read_csv(file_path)

# Case 1: Analyzing the performance of dairy farms
farm_performance = df.groupby(['Location', 'Total Land Area (acres)', 'Number of Cows']).agg({
    'Total Value': 'sum'
}).reset_index()
top_performing_farms = farm_performance.sort_values(by='Total Value', ascending=False).head(10)
print("\nTop-performing farms:")
print(top_performing_farms)

# Case 2: Sales and distribution patterns of dairy products
product_distribution = df.groupby(['Product Name', 'Brand', 'Customer Location']).agg({
    'Quantity Sold (liters/kg)': 'sum',
    'Approx. Total Revenue(INR)': 'sum'
}).reset_index()
top_selling_products = product_distribution.sort_values(by='Quantity Sold (liters/kg)', ascending=False).head(10)
print("\nTop-selling products:")
print(top_selling_products)

# Case 3: Impact of storage conditions and shelf life
sns.scatterplot(x='Shelf Life (days)', y='Storage Condition', hue='Quantity Sold (liters/kg)', data=df)
plt.title('Impact of Storage Conditions and Shelf Life on Quantity Sold')
plt.show()

# Case 4: Customer preferences and buying behavior
customer_behavior = df.groupby(['Location', 'Sales Channel']).agg({
    'Quantity Sold (liters/kg)': 'sum',
    'Approx. Total Revenue(INR)': 'sum'
}).reset_index()
print("\nCustomer Behavior:")
print(customer_behavior)

# Case 5: Inventory management
inventory_management = df.groupby(['Product Name']).agg({
    'Quantity in Stock (liters/kg)': 'sum',
    'Minimum Stock Threshold (liters/kg)': 'mean',
    'Reorder Quantity (liters/kg)': 'mean'
}).reset_index()
print("\nInventory Management:")
print(inventory_management)

# Case 6: Market research and trend analysis
market_trends = df.groupby(['Date']).agg({
    'Quantity Sold (liters/kg)': 'sum',
    'Approx. Total Revenue(INR)': 'sum'
}).reset_index()
plt.figure(figsize=(10, 6))
plt.plot(market_trends['Date'], market_trends['Quantity Sold (liters/kg)'], label='Quantity Sold')
plt.plot(market_trends['Date'], market_trends['Approx. Total Revenue(INR)'], label='Total Revenue')
plt.xlabel('Date')
plt.ylabel('Quantity/Revenue')
plt.title('Market Trends')
plt.legend()
plt.show()

# Case 7: Predictive models (Placeholder for actual code, you may use machine learning libraries)
# Example: Train a regression model to predict future sales or revenue

# End of script
