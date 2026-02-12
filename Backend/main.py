import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Create the data from the image
# We represent the intervals in descending order as seen in the image.
data = {
    'Scores': ['51-60', '41-50', '31-40', '21-30', '11-20', '1-10'],
    'f': [4, 7, 10, 8, 6, 5],
    'x': [55.5, 45.5, 35.5, 25.5, 15.5, 5.5],
    'fx': [222, 318.5, 355, 204, 93, 27.5]
}

df = pd.DataFrame(data)

# Sort the data by the score intervals (lowest to highest)
# To sort correctly, we can extract the start of the range
df['start_range'] = df['Scores'].str.split('-').str[0].astype(int)
df = df.sort_values(by='start_range').reset_index(drop=True)

# 2. Handle Error/Missing Data (Simulated check)
# Recalculate fx to ensure accuracy and check for nulls
df['x_calc'] = df['Scores'].apply(lambda s: sum(map(int, s.split('-'))) / 2.0)
df['fx_calc'] = df['f'] * df['x']

# If there were missing values, we could use:
# df['f'] = df['f'].fillna(df['f'].mean())
# etc.

# 3. Statistical Calculations
sum_f = df['f'].sum()
sum_fx = df['fx'].sum()
mean_val = sum_fx / sum_f

# Median for grouped data
# Median class is where cumulative frequency reaches sum_f/2
df['cf'] = df['f'].cumsum()
median_idx = df[df['cf'] >= sum_f / 2].index[0]
median_class = df.iloc[median_idx]
L = float(median_class['Scores'].split('-')[0]) - 0.5 # lower boundary
cf_prev = df.iloc[median_idx - 1]['cf'] if median_idx > 0 else 0
f_median = median_class['f']
h = 10 # class width
median_val = L + ((sum_f/2 - cf_prev) / f_median) * h

# Mode for grouped data
modal_idx = df['f'].idxmax()
modal_class = df.iloc[modal_idx]
L_mode = float(modal_class['Scores'].split('-')[0]) - 0.5
f1 = modal_class['f']
f0 = df.iloc[modal_idx - 1]['f'] if modal_idx > 0 else 0
f2 = df.iloc[modal_idx + 1]['f'] if modal_idx < len(df)-1 else 0
mode_val = L_mode + ((f1 - f0) / (2*f1 - f0 - f2)) * h

# 4. Graphs
# Bar Chart
plt.bar(df['Scores'], df['f'], color='skyblue', edgecolor='black')
plt.title('Frequency of Score Intervals')
plt.xlabel('Scores')
plt.ylabel('Frequency (f)')
plt.savefig('bar_chart.png')
plt.close()

# Pie Chart
plt.pie(df['f'], labels=df['Scores'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Distribution of Scores')
plt.savefig('pie_chart.png')
plt.close()

# Save cleaned data to CSV
df[['Scores', 'f', 'x', 'fx']].to_csv('processed_scores.csv', index=False)

print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
print(df[['Scores', 'f', 'x', 'fx']])