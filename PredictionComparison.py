import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/SEMESTER4/PROJECTWORK/PREDICTIONCOMPARISON.csv')

plt.figure(figsize=(10, 6))

# Plot Actual Lead Time against Product Names (X: Lead Time, Y: Product Name)
plt.plot(df['Actual_Lead_Time'], df['Product_Name'], marker='o', linestyle='dashed', label='Actual Lead Time')

# Plot Predicted Lead Time against Product Names (X: Lead Time, Y: Product Name)
plt.plot(df['Predicted_Lead_Time'], df['Product_Name'], marker='x', linestyle='dashed', label='Predicted Lead Time')

plt.xlabel('Lead Time')
plt.ylabel('Product Name')
plt.title('Prediction Comparison by Product')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()