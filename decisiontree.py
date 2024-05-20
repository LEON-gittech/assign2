import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib
from sklearn.preprocessing import StandardScaler
matplotlib.use('TkAgg')  # 指定使用 TkAgg 后端
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
# 加载数据
df = pd.read_csv("df_cleaned_83.csv")

# 假设在此之前已经删除了某些不需要的列
categorical_cols = ['Species', 'Population', 'Thorax_length', 'wing_loading', 'Sex']
X = df.drop(columns=categorical_cols)
y = df['Sex']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


train_accuracies = []
test_accuracies = []


def evaluate_decision_tree(max_depth_options):
       # 遍历 max_features
    for max_depth in max_depth_options:
        dt_classifier = DecisionTreeClassifier(max_depth = max_depth)
        dt_classifier.fit(X_train, y_train)

        y_train_pred = dt_classifier.predict(X_train)
        y_test_pred = dt_classifier.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        max_accuracy = np.max(test_accuracies)
        print(f"Processed max_depth = {max_depth}")
        print(f"normal best test accuracy without Gridsearch = {max_accuracy}")
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix for original decision tree')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    # 绘制结果
    plt.figure(figsize=(10, 5))
    plt.plot(max_depth_options, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(max_depth_options, test_accuracies, label='Test Accuracy', marker='o')
    plt.title('Decision Tree Accuracy vs. max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.xticks(max_depth_options, labels=[str(option) for option in max_depth_options])  # 设置x轴标签
    plt.legend()
    plt.grid(True)
    plt.show()

    # 找出最高测试准确率及对应的 max_depth 值
    max_accuracy_index = test_accuracies.index(max(test_accuracies))
    max_accuracy = test_accuracies[max_accuracy_index]
    max_accuracy_depth = max_depth_options[max_accuracy_index]

    print(f"Best Test Accuracy: {max_accuracy} at max_depth = {max_accuracy_depth}")




# 测试函数
# max_depth_options = [4]
# evaluate_decision_tree(max_depth_options)


def optimize_and_visualize_decision_tree():
    # 创建决策树模型
    dt_classifier = DecisionTreeClassifier()

    # 创建 Pipeline，这里可以加入其他预处理步骤
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 可选，根据需要决定是否标准化
        ('classifier', dt_classifier)
    ])

    # 设置网格搜索的参数范围
    param_grid = {
        'classifier__max_depth': [None, 3, 5, 10, 15],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy']
    }

    # 创建网格搜索对象，使用交叉验证
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

    # 进行网格搜索
    grid_search.fit(X_train, y_train)

    # 找到最优参数和最优模型
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 在测试集上评估最优模型
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print("Best parameters found: ", best_params)
    print("Best Test Accuracy: ", test_accuracy)

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix for optimized parameter')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # 可视化准确率的折线图
    plt.figure(figsize=(10, 5))
    plt.plot(grid_search.cv_results_['mean_test_score'], label='Mean CV Accuracy', marker='o')
    plt.plot([train_accuracy] * len(grid_search.cv_results_['mean_test_score']), label='Train Accuracy', linestyle='--')
    plt.plot([test_accuracy] * len(grid_search.cv_results_['mean_test_score']), label='Test Accuracy', linestyle='--')
    plt.title('Decision Tree Model Accuracy')
    plt.xlabel('Parameter Setting Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


    # 可视化结果
    results = pd.DataFrame(grid_search.cv_results_)
    pivot_table = results.pivot_table(values='mean_test_score',
                                      index='param_classifier__max_depth',
                                      columns='param_classifier__min_samples_split')

    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".3f")
    plt.title('Decision Tree Performance Heatmap')
    plt.xlabel('Min Samples Split')
    plt.ylabel('Max Depth')
    plt.show()
optimize_and_visualize_decision_tree()
