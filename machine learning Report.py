import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv")
# 假设 df 是你的原始数据框架
# 假设 rows_to_remove 包含了所有异常值行的索引
rows_to_remove = [61, 698]
# 删除这些行
df_cleaned = df.drop(index=rows_to_remove)
# 现在 df_cleaned 是删除了异常值的数据集
# print(f"{df_cleaned.head()}\n")
# print(f"---")
print(f"{df_cleaned.info()}\n")
# print(f"------")
print(f"{df_cleaned.isnull().sum()}\n")
# print(f"-----------")
# Boxplots for all numerical columns
def plot_boxplots(dataframe):
    num_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
    dataframe[num_columns].plot(kind='box', subplots=True, layout=(len(num_columns)//3+1, 3), figsize=(20, 20), fontsize=12)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
#
# plot_boxplots(df_cleaned)
#
# Z-scores to identify outliers
def identify_outliers_with_zscore(dataframe, threshold=3):
    num_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
    z_scores = np.abs(stats.zscore(dataframe[num_columns]))
    outliers = (z_scores > threshold).any(axis=1)
    print("Outliers found: \n", dataframe[num_columns][outliers])
#
# identify_outliers_with_zscore(df_cleaned)

# 选择数值型特征
# num_columns = df_cleaned.select_dtypes(include=['int64']).columns
#
# # 使用 seaborn 的 pairplot 函数来创建散点图矩阵
# sns.pairplot(df_cleaned[num_columns])
# plt.show()

def haversine(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # 哈弗辛公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # 地球平均半径，单位公里
    r = 6371
    return c * r
# 假设你有两列分别代表纬度和经度
# df_cleaned['Latitude'], df_cleaned['Longitude']
# 你可以计算数据集中每对坐标之间的距离，例如计算第一个点与所有其他点的距离
first_lat = df_cleaned['Latitude'].iloc[0]
first_lon = df_cleaned['Longitude'].iloc[0]

df_cleaned['distance_from_first'] = [
    haversine(first_lat, first_lon, lat, lon) for lat, lon in zip(df_cleaned['Latitude'], df_cleaned['Longitude'])
]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_scatter(dataframe, lat_col, lon_col, dist_col):
    """
    绘制一个基于纬度、经度和距离的3D散点图。

    参数:
    - dataframe: 包含数据的pandas DataFrame。
    - lat_col: 纬度列的名称。
    - lon_col: 经度列的名称。
    - dist_col: 距离列的名称。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取数据
    latitudes = dataframe[lat_col]
    longitudes = dataframe[lon_col]
    distances = dataframe[dist_col]

    # 散点图
    sc = ax.scatter(latitudes, longitudes, distances, c=distances, cmap='viridis')

    # 标签和标题
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Distance from First Point')
    ax.set_title('3D Scatter Plot of Geographic Data')

    # 颜色条
    plt.colorbar(sc)

    plt.show()

plot_3d_scatter(df_cleaned, 'Latitude', 'Longitude', 'distance_from_first')
# 保存 DataFrame 到 CSV
df_cleaned.to_csv('df_cleaned_83.csv', index=False)


