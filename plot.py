import numpy as np
import pandas
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

data = pandas.read_csv(
    'newdata.csv',
    names=['Count', 'steering']
)

# data = pandas.read_csv(
#     'data/driving_log.csv',
#     names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
# )

data = shuffle(data)

# x = data['steering'].head().round(1)
# y = data['steering'].head().round(1).value_counts(normalize=True)
# print(x)
# print(y)
# x = np.arange(0, 5, 0.1);
# y = np.sin(x)
#data['steering'].head().round(1).plot(kind='bar')

data.round(1).groupby(['steering']).count().plot(kind='bar')
#data['steering'].head().round(1).count().plot()

plt.show()