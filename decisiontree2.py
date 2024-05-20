import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 加载数据
df = pd.read_csv("df_cleaned_83.csv")
print(df.info())

# 定义分类列
categorical_cols = ['Species', 'Population', 'Thorax_length', 'wing_loading']

# 使用 OneHotEncoder 对分类特征进行编码
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'  # 保留其他非分类列
)

# 准备特征和目标变量
X = df.drop(columns=['Sex'])  # 假设我们要预测 'Sex' 列
y = df['Sex']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 应用 OneHotEncoder
X_train = column_transformer.fit_transform(X_train)
X_test = column_transformer.transform(X_test)

# 创建决策树分类器实例
dt_classifier = DecisionTreeClassifier()

# 训练模型
dt_classifier.fit(X_train, y_train)

# 对训练集和测试集进行预测
y_train_pred = dt_classifier.predict(X_train)
y_test_pred = dt_classifier.predict(X_test)

# 获取预测概率用于 log loss 计算
y_train_proba = dt_classifier.predict_proba(X_train)
y_test_proba = dt_classifier.predict_proba(X_test)

# 计算准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 计算训练和测试集的 log loss
train_loss = log_loss(y_train, y_train_proba)
test_loss = log_loss(y_test, y_test_proba)

# 打印准确率和损失
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Train Loss: {train_loss}")
print(f"Test Loss: {test_loss}")
