import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Date': ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'],
    'Close_Price': [2800.00, 2820.00, 2830.00, 2840.00, 2860.00]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
start_date = '2023-01-02'
end_date = '2023-01-06'
filtered_data = df[start_date:end_date]
plt.figure(figsize=(12, 6))
plt.plot(filtered_data.index, filtered_data['Close_Price'], marker='o', linestyle='-', color='b')
plt.title('Alphabet Inc. Stock Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.grid(True)
plt.show()
