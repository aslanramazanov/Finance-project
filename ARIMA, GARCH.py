import numpy as np
from IPython.display import display
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
import arch


path = "/Users/aslanramazanov/Desktop/data_excel.xlsx"
data = pd.ExcelFile(path).parse()
dataset = pd.DataFrame(data)
display(dataset.head(0))

date = np.array(dataset['Date'])
returns = dataset['ln_AAPL_ret']

timeseries = np.array(returns)
result = adfuller(timeseries)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
#p-value is greater that significance level -> there is autocorrelation in the data. Starting to difference it

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(date, timeseries); axes[0, 0].set_title('Original Series')
plot_acf(timeseries, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(np.diff(timeseries)); axes[1, 0].set_title('1st Order Differencing')
plot_acf(np.diff(timeseries), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(np.diff(np.diff(timeseries))); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(np.diff(np.diff(timeseries)), ax=axes[2, 1])

plt.show()

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

#partial autocorrelation of 1st differenced timeseries
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(np.diff(timeseries)); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(np.diff(timeseries), ax=axes[1])

plt.show()
#из графика выбираем параметр р для AR равным 0


#autocorrelation plot of 1st differenced timeseries
from statsmodels.graphics.tsaplots import plot_acf
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(np.diff(timeseries)); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(np.diff(timeseries), ax=axes[1])

plt.show()

#seems like that the timeseries is over-differenced, so I'll add 1 MA term


from statsmodels.tsa.arima.model import ARIMA

#ARIMA Model
model = ARIMA(timeseries, order=(1,2,2)) #p,d,q parameters
model_fit = model.fit()
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

#Построим теперь GARCH модель

garch = arch.arch_model(returns, p=1, q=1)
fgarch = garch.fit(disp='off')
print(fgarch.summary())
#наилучший результат по АИК достигается при параметрах равных единице  При их увеличении модель становится хуже

