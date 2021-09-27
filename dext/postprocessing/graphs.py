import pandas as pd
import matplotlib.pyplot as plt

ap_curve = pd.read_excel('images/results/report.xlsx', sheet_name='ap_curve')
ap_curve.loc["mean"] = ap_curve.iloc[:, 3:].mean(axis=0)
columns = ap_curve.columns.to_numpy()
ap_percent50 = (ap_curve.filter(regex='ap_50percent_'))
mean_ap_curve = ap_percent50.iloc[-1, :].to_numpy()
range_values = [float(val.split('percent_')[1])
                for val in ap_percent50.columns]
fig = plt.figure()
plt.plot(range_values, mean_ap_curve)
plt.xlabel('Fraction of pixels flipped')
plt.ylabel('AP @[IOU=0.50]')
plt.title('Pixel flipped vs AP')
fig.savefig('mean_ap.jpg')
