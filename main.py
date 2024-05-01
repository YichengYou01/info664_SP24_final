import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import folium


# 定义函数来获取所有页面的数据
def get_all_data(api_url):
    all_data = pd.DataFrame()  # 创建一个空的DataFrame来存储所有数据
    limit = 50000  # 每一页的记录数限制

    offset = 0  # 初始偏移量
    while True:
        # 构建API请求URL
        url = f"{api_url}?$limit={limit}&$offset={offset}"

        # 从API获取数据
        data = pd.read_json(url)

        # 将当前页的数据添加到总数据中
        all_data = pd.concat([all_data, data], ignore_index=True)

        # 如果当前页的数据少于限制数量，说明已经获取完所有数据，退出循环
        if len(data) < limit:
            break

        # 更新偏移量，准备获取下一页数据
        offset += limit

    return all_data


# 定义API URL
api_url = "https://data.cityofnewyork.us/resource/5rq2-4hqu.json"

# 获取所有数据
source_data = get_all_data(api_url)
row_count = len(source_data)
print("source_data的行数:", row_count)

# 只选择health为Good的树木作为训练集，并随机抽样其中的百分之10
data = source_data[source_data['health'] == 'Good'].sample(frac=0.1, random_state=42).copy()

# 数据预处理
# 处理steward字段
data['steward'] = data['steward'].apply(lambda x: 0 if x == 'None' else int(x.split('or')[0]))

interested_columns = ['tree_dbh', 'curb_loc', 'status', 'health', 'spc_common', 'guards', 'sidewalk', 'problems', 'steward']

# 仅保留感兴趣的字段作为特征
data = data[interested_columns]

# 特征编码
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# 划分数据集
X = data.drop(columns=['steward'])  # 移除目标变量
y = data['steward']  # 以steward作为目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 模型训练
# model = LogisticRegression()
# model.fit(X_train, y_train)
# 构建随机森林模型
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, max_features=None, criterion='gini')

# 训练模型
random_forest_model.fit(X_train, y_train)

# 模型评估
accuracy = random_forest_model.score(X_test, y_test)
print("随机森林模型准确率:", accuracy)

# 数据预处理
# 处理steward字段
source_data['steward'] = source_data['steward'].apply(lambda x: 0 if x == 'None' or pd.isna(x) else int(x.split('or')[0]))

# 获取原始数据中 管理人员 0 1 2 ...的数量
steward_count = source_data['steward'].value_counts()
print("原始数据中 管理人员 0 1 2 ...的数量:", steward_count)

# 选择感兴趣的字段
interested_columns = ['tree_dbh', 'curb_loc', 'status', 'health', 'spc_common', 'guards', 'sidewalk', 'problems']

# 仅保留感兴趣的字段作为特征
new_data = source_data[interested_columns].copy()

# 特征编码
le = LabelEncoder()
for col in new_data.columns:
    if new_data[col].dtype == 'object':
        new_data[col] = le.fit_transform(new_data[col])

# 使用训练好的模型进行预测
predicted_steward = random_forest_model.predict(new_data)
print("预测的管理人数:", predicted_steward)
# 统计预测的管理人数中 0 1 2 ...的数量
predicted_steward_count = pd.Series(predicted_steward).value_counts()
print("预测的管理人数中 0 1 2 ...的数量:", predicted_steward_count)

# 计算实际的管理人数
actual_steward = source_data['steward']

# 计算管理人数差异
steward_difference = actual_steward - predicted_steward

# 计算管理人数差异的百分比
percentage_difference = (steward_difference / actual_steward) * 100

# 计算多了和少了的数量
surplus_count = len(percentage_difference[percentage_difference > 0])
deficit_count = len(percentage_difference[percentage_difference < 0])
equal_count = len(percentage_difference[percentage_difference == 0])

# 制作饼图
labels = ['Surplus', 'Deficit', 'Equal']
sizes = [surplus_count, deficit_count, equal_count]
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.1, 0, 0)  # 突出显示多了的部分

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Percentage of Steward Surplus, Deficit, and Equal')
plt.show()

# 创建一个基于纽约市中心的地图
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

# 根据数据中的位置信息和管理人数情况添加标记
for index, row in source_data.iterrows():
    color = ''
    if percentage_difference[index] > 0:
        color = 'green'  # 管理人数多余
    elif percentage_difference[index] < 0:
        color = 'red'  # 管理人数不足
    else:
        color = 'blue'  # 管理人数相等
    folium.CircleMarker(location=[row['latitude'], row['longitude']],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color).add_to(nyc_map)

# 保存地图为HTML文件
nyc_map.save('steward_map.html')
