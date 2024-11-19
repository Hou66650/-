import re
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns


class TestCleaner:

    @classmethod
    def percentage_to_decimal(cls, df, column_name):
        """
        将数据框中指定列的百分数字符串转换为小数。

        参数:
        df (pd.DataFrame): 数据框对象
        column_name (str): 要转换的列名

        返回:
        pd.DataFrame: 转换后的数据框
        """
        # 使用apply函数将百分数字符串转换为小数
        df[column_name] = df[column_name].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x)
        return df

    @classmethod
    def split_column_and_rename(cls, df, column_name, new_column_name1, new_column_name2):
        """
        将数据框中指定列的特定格式（如1600(1:0.5)）拆分为两列，并分别重新命名这些列。

        参数:
        df (pd.DataFrame): 数据框对象
        column_name (str): 要拆分的列名
        new_column_name1 (str): 拆分后的第一列的新名称
        new_column_name2 (str): 拆分后的第二列的新名称

        返回:
        pd.DataFrame: 拆分并重命名后的数据框
        """
        # 拆分列并创建新的数据框
        split_df = df[column_name].str.extract(r'(\d+)\(1:([\d.]+)\)', expand=True)

        # 重命名新列
        split_df.columns = [new_column_name1, new_column_name2]

        # 将新列添加到原始数据框
        df = pd.concat([df, split_df], axis=1)

        # 删除原来的列
        df = df.drop(column_name, axis=1)

        return df

    @classmethod
    def convert_columns_to_float(cls, df, column_names):
        """
        将数据框中指定列的字符串转换为浮点数。

        参数:
        df (pd.DataFrame): 数据框对象
        column_names (list): 要转换的列名列表

        返回:
        pd.DataFrame: 转换后的数据框
        """
        # 遍历每个列名，并尝试将字符串转换为浮点数
        for column_name in column_names:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df

    @classmethod
    def filter_correlation_matrix_and_df(cls, df, target_column, threshold):
        """
        基于某个列名筛选出相关性小于某个阈值的列，并从相关性矩阵和原始数据框中删除这些列。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名
        threshold (float): 相关性阈值

        返回:
        tuple: 包含两个元素 (filtered_correlation_matrix, filtered_df)
            - filtered_correlation_matrix: 筛选后的相关性矩阵
            - filtered_df: 筛选后的原始数据框
        """
        # 计算相关性矩阵
        correlation_matrix = df.corr()

        # 筛选出相关性小于阈值的列
        columns_to_drop = correlation_matrix.index[correlation_matrix[target_column] < threshold]

        # 删除这些列
        filtered_correlation_matrix = correlation_matrix.drop(columns_to_drop, axis=1).drop(columns_to_drop, axis=0)

        # 筛选出剩余列的原始数据框
        filtered_df = df.drop(columns_to_drop, axis=1)

        return filtered_correlation_matrix, filtered_df

    @classmethod
    def train_and_predict_with_random_forest(cls, df, target_column):
        """
        使用随机森林回归模型对数据框中的一列进行预测，并处理包含 NaN 的数据。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名

        返回:
        tuple: 包含两个元素 (predictions, mse)
            - predictions: 预测值
            - mse: 均方误差
        """
        # 将目标列与其他列分离
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X = imputer.fit_transform(X)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # 初始化随机森林回归模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 进行预测
        predictions = model.predict(X_test)

        # 计算均方误差
        mse = mean_squared_error(y_test, predictions)

        return predictions, mse

    @classmethod
    def predict_and_append_to_df(cls, df, model_filename, target_column, new_column_name='Predicted',
                                 impute_strategy='mean'):
        """
        加载保存的模型并使用新数据进行预测，然后将预测结果作为新的一列添加到数据框中。

        参数:
        df (pd.DataFrame): 数据框对象
        model_filename (str): 保存模型的文件名
        target_column (str): 目标列名
        new_column_name (str): 新添加的预测列的名称
        impute_strategy (str): SimpleImputer 的填充策略，默认为 'mean'

        返回:
        pd.DataFrame: 包含预测结果的数据框
        """
        # 加载模型
        model = joblib.load(model_filename, mmap_mode='r')  # 使用 mmap_mode 可以减少内存使用

        # 将目标列与其他列分离
        X_new = df.drop(columns=[target_column])

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy=impute_strategy)
        X_new = imputer.fit_transform(X_new)

        # 进行预测
        predictions = model.predict(X_new)

        # 将预测结果添加到数据框中
        df[new_column_name] = predictions

        return df

    @classmethod
    def calculate_mean_by_category(cls, df, category_column, value_columns):
        """
        根据某一列的值进行分类，对另外两列的值求均值，并存储到新的数据框中。

        参数:
        df (pd.DataFrame): 原始数据框
        category_column (str): 用于分类的列名
        value_columns (list): 需要求均值的列名列表

        返回:
        pd.DataFrame: 包含分类和均值结果的数据框
        """
        # 根据分类列进行分组，并对指定列求均值
        grouped = df.groupby(category_column)[value_columns].mean().reset_index()

        # 重命名列名，以便更清晰地表示均值
        grouped.columns = [category_column] + [f'Mean_{col}' for col in value_columns]

        return grouped

    @classmethod
    def clean_and_convert_column(cls, df, column_name):
        """
        去除指定列中的非法字符并转换为浮点数。

        参数:
        df (pd.DataFrame): 原始数据框
        column_name (str): 需要处理的列名

        返回:
        pd.DataFrame: 处理后的数据框
        """
        # 使用正则表达式去除非数字字符
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\d.]', '', str(x)))

        # 尝试将清洗后的数据转换为浮点数
        try:
            df[column_name] = df[column_name].astype(float)
        except ValueError as e:
            print(f"Error converting column {column_name} to float: {e}")

        return df

    @classmethod
    def plot_coordinates_and_heatmap(cls, df):
        """
        根据df中的X,Y数据标出具体坐标图，并以此根据adcode和相应的price绘制价格地区热力图。

        参数:
        df (pd.DataFrame): 包含X, Y, adcode和price列的数据框
        """
        # 绘制坐标图
        plt.figure(figsize=(10, 6))
        plt.scatter(df['X'], df['Y'], c=df['Total number of households'], cmap='YlOrRd', s=10)
        plt.colorbar(label='Total number of households')
        plt.xlabel('X (Longitude)')
        plt.ylabel('Y (Latitude)')
        plt.title('Coordinates and Person_num Heatmap')
        plt.grid(True)
        plt.show()

        # 根据adcode和price绘制价格地区热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.pivot_table(index='adcode', values='Total number of households', aggfunc='mean'), cmap='hot',
                    annot=True,
                    fmt='.0f')
        plt.xlabel('Adcode')
        plt.ylabel('Adcode')
        plt.title('Person_num Heatmap by Adcode')
        plt.show()

    @classmethod
    def plot_coordinates_and_heatmap_p(cls, df):
        """
        根据df中的X,Y数据标出具体坐标图，并以此根据adcode和相应的price绘制价格地区热力图。
        同时绘制一个热力扩散图和一个辐射图。

        参数:
        df (pd.DataFrame): 包含X, Y, adcode和price列的数据框
        """
        # 绘制坐标图
        plt.figure(figsize=(10, 6))
        plt.scatter(df['X'], df['Y'], c=df['Total number of households'], cmap='hot', s=5)  # 使用 'hot' 颜色映射
        plt.colorbar(label='Total number of households')
        plt.xlabel('X (Longitude)')
        plt.ylabel('Y (Latitude)')
        plt.title('Coordinates and Person_num Heatmap')
        plt.grid(True)
        plt.show()

        # 根据adcode和price绘制价格地区热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.pivot_table(index='adcode', values='Total number of households', aggfunc='mean'), cmap='hot',
                    annot=True,
                    fmt='.0f')  # 使用 'hot' 颜色映射
        plt.xlabel('Adcode')
        plt.ylabel('Adcode')
        plt.title('Person_num Heatmap by Adcode')
        plt.show()

        # 绘制热力扩散图
        plt.figure(figsize=(10, 6))
        xy = df[['X', 'Y']].values
        xy_kde = gaussian_kde(xy.T)(xy.T)
        plt.scatter(df['X'], df['Y'], c=xy_kde, cmap='viridis', s=5)  # 使用 'viridis' 颜色映射
        plt.colorbar(label='Density')
        plt.xlabel('X (Longitude)')
        plt.ylabel('Y (Latitude)')
        plt.title('Heatmap Diffusion')
        plt.grid(True)
        plt.show()

        # 绘制辐射图
        plt.figure(figsize=(8, 8))
        labels = df['adcode'].unique()
        price_means = [df[df['adcode'] == adcode]['Total number of households'].mean() for adcode in labels]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        price_means += price_means[:1]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.plot(angles, price_means, 'o-', linewidth=2)
        ax.fill(angles, price_means, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        plt.yticks([500, 1000, 1500, 20000, 3500], [500, 1000, 1500, 20000, 3500], color="grey", size=7)
        plt.ylim(0, 4000)
        plt.title('Radar Chart of Mean Person_num by Adcode')
        plt.grid(True)
        plt.show()

    @classmethod
    def plot_na_num(cls, df, title_name):
        # 统计每列的缺失值数量
        missing_values = df.isnull().sum()

        # 计算缺失值占比
        missing_percentage = (missing_values / len(df)) * 100

        # 将结果展示为DataFrame
        missing_info = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Missing Percentage (%)': missing_percentage.values
        })

        # 显示缺失值统计信息
        print(missing_info)

        # 绘制条形图
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Column', y='Missing Percentage (%)', data=missing_info)
        plt.xticks(rotation=90)
        plt.title(title_name)
        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage (%)')
        plt.show()

    @classmethod
    def visualize_distribution(cls, df, column_name):
        # 检查列是否存在
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in the DataFrame.")
            return

        # 创建子图
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制直方图
        sns.histplot(df[column_name], kde=True, color='blue', ax=ax)

        # 设置标题和标签
        ax.set_title(f'Distribution of {column_name}', fontsize=16)
        ax.set_xlabel(column_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        # 显示图形
        plt.show()


class ModelTrain:
    def __init__(self, model_name):
        self.model_name = model_name

    def train_save(self, df, target_column, model_filename):
        if self.model_name == 'random_forest':
            return self._train_save_random_forest(df, target_column, model_filename)
        elif self.model_name == 'GBDT':
            return self._train_save_gbdt(df, target_column, model_filename)
        elif self.model_name == 'xgboost':
            return self._train_save_xgboost(df, target_column, model_filename)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _train_save_random_forest(self, df, target_column, model_filename):
        """
        使用随机森林回归模型对数据框中的一列进行预测，并处理包含 NaN 的数据。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名
        model_filename (str): 模型文件名

        返回:
        tuple: 包含三个元素 (predictions, adjusted_r2, fig)
            - predictions: 预测值
            - adjusted_r2: 调整后的 R² 值
            - fig: 可视化图表对象
        """
        # 将目标列与其他列分离
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X = imputer.fit_transform(X)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # 初始化随机森林回归模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 进行预测
        predictions = model.predict(X_test)

        # 计算 R² 值
        r2 = r2_score(y_test, predictions)

        # 计算调整后的 R² 值
        n = X_test.shape[0]  # 样本数量
        p = X_test.shape[1]  # 特征数量
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 保存模型
        joblib.dump(model, model_filename)

        # 可视化训练过程中的预测值与实际值
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions, ax=ax, label='Predicted vs Actual')
        sns.lineplot(x=y_test, y=y_test, color='red', ax=ax, label='Perfect Fit')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()

        # 显示 R² 值和调整后的 R² 值
        ax.text(0.05, 0.95, f'R²: {r2:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        ax.text(0.05, 0.90, f'Adjusted R²: {adjusted_r2:.4f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        plt.tight_layout()
        plt.show()

        return predictions, adjusted_r2, fig

    def _train_save_gbdt(self, df, target_column, model_filename):
        """
        使用梯度提升回归树模型对数据框中的一列进行预测，并处理包含 NaN 的数据。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名

        返回:
        tuple: 包含两个元素 (predictions, adjusted_r2)
            - predictions: 预测值
            - adjusted_r2: 调整后的 R² 值
        """
        # 将目标列与其他列分离
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X = imputer.fit_transform(X)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # 初始化梯度提升回归树模型
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 进行预测
        predictions = model.predict(X_test)

        # 计算 R² 值
        r2 = r2_score(y_test, predictions)

        # 计算调整后的 R² 值
        n = X_test.shape[0]  # 样本数量
        p = X_test.shape[1]  # 特征数量
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 保存模型
        joblib.dump(model, model_filename)

        return predictions, adjusted_r2

    def _train_save_xgboost(self, df, target_column, model_filename):
        """
        使用XGBoost模型对数据框中的一列进行预测，并处理包含 NaN 的数据。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名
        model_filename (str): 保存模型的文件名

        返回:
        tuple: 包含两个元素 (predictions, adjusted_r2)
            - predictions: 预测值
            - adjusted_r2: 调整后的 R² 值
        """
        # 将目标列与其他列分离
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X = imputer.fit_transform(X)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # 初始化XGBoost回归模型
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3,
                                 random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 进行预测
        predictions = model.predict(X_test)

        # 计算 R² 值
        r2 = r2_score(y_test, predictions)

        # 计算调整后的 R² 值
        n = X_test.shape[0]  # 样本数量
        p = X_test.shape[1]  # 特征数量
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 保存模型
        joblib.dump(model, model_filename)

        return predictions, adjusted_r2

    def predict_with_saved_model(self, new_df, model_filename, target_column):
        if self.model_name == 'random_forest' or 'GBDT' or 'xgboost':
            return self._predict_with_saved_random_forest(new_df, model_filename, target_column)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _predict_with_saved_random_forest(self, new_df, model_filename, target_column):
        """
               加载保存的模型并使用新数据进行预测。

               参数:
               new_df (pd.DataFrame): 新数据框对象
               model_filename (str): 保存模型的文件名
               target_column (str): 目标列名

               返回:
               predictions: 预测值
               """
        # 加载模型
        model = joblib.load(model_filename)

        # 将目标列与其他列分离
        X_new = new_df.drop(columns=[target_column])

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X_new = imputer.fit_transform(X_new)

        # 进行预测
        predictions = model.predict(X_new)

        return predictions


class SimplifiedDeepNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplifiedDeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_save_pytorch(df, target_column, model_filename, input_size, hidden_size=128, output_size=1, num_epochs=100,
                       batch_size=16, learning_rate=0.01):
    """
    使用PyTorch搭建并训练一个复杂的神经网络模型，对数据框中的一列进行预测，并处理包含NaN的数据。

    参数:
    df (pd.DataFrame): 数据框对象
    target_column (str): 目标列的名称
    model_filename (str): 保存模型的文件名
    input_size (int): 输入特征的数量
    hidden_size (int): 隐藏层的大小
    output_size (int): 输出特征的数量
    num_epochs (int): 训练的轮数
    batch_size (int): 批量大小
    learning_rate (float): 学习率
    """
    # 处理NaN值
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # 分割数据集
    X = df_imputed.drop(columns=[target_column])
    y = df_imputed[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = SimplifiedDeepNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化损失列表
    train_losses = []
    val_losses = []

    # 实时显示曲线图
    plt.figure()
    plt.ion()  # 开启交互模式

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 每个epoch结束后，记录训练损失和验证损失
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in test_loader:
                y_val_pred = model(X_val_batch)
                val_loss += criterion(y_val_pred, y_val_batch).item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        # 更新曲线图
        plt.clf()
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
        plt.plot(range(1, epoch + 2), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.draw()
        plt.pause(0.01)

        # 打印训练损失和验证损失
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # 保存模型
    torch.save(model.state_dict(), model_filename)

    # 加载模型并进行评估
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred = model(X_batch)
            y_pred_list.append(y_pred)

    y_pred_tensor = torch.cat(y_pred_list, dim=0)
    test_r2 = r2_score(y_test_tensor.numpy(), y_pred_tensor.numpy())
    print(f'Test R^2 Score: {test_r2}')

    # 关闭交互模式
    plt.ioff()
    plt.show()


class DataProcessor:
    def __init__(self, process_name):
        """
        初始化类实例。

        参数:
        df (pd.DataFrame): 输入的DataFrame。
        columns (list): 需要处理的列名列表。
        process_name (str): 处理的名称，如 'na20' 表示将缺失值替换为0。
        'na2mean':把na值替换成均值
        '%2dot':把百分数转化为小数
        ‘na2’
        """
        self.process_name = process_name

    def fill_na_with_zero(self, df, columns):
        """
        将指定列中的缺失值替换为0。

        参数:
        df (pd.DataFrame): 输入的DataFrame。
        columns (list): 需要处理的列名列表。

        返回:
        pd.DataFrame: 处理后的DataFrame。
        """
        for col in columns:
            df[col] = df[col].fillna(0)
        return df

    def fill_na_with_mean(self, df, columns):
        """
        将指定列中的缺失值替换为该列其他值的均值。

        参数:
        df (pd.DataFrame): 输入的DataFrame。
        columns (list): 需要处理的列名列表。

        返回:
        pd.DataFrame: 处理后的DataFrame。
        """
        for col in columns:
            mean_value = df[col].mean(skipna=True)  # 计算该列的均值，忽略缺失值
            df[col] = df[col].fillna(mean_value)  # 将缺失值替换为均值
        return df

    def convert_percent_to_decimal(self, df, column_name):
        """
        将指定列中的百分数转换为小数形式。

        参数:
        dataframe (pd.DataFrame): 包含数据的DataFrame。
        column_name (str): 要转换的列名。

        返回:
        pd.DataFrame: 转换后的DataFrame。
        """
        # 检查列名是否存在于DataFrame中
        if column_name not in df.columns:
            raise ValueError(f"列名 {column_name} 不存在于DataFrame中。")

        # 将百分数转换为小数
        df[column_name] = df[column_name].apply(
            lambda x: float(x.rstrip('%')) / 100 if isinstance(x, str) else x)

        return df

    def fill_na_with_specific_string(self, df, column_names):
        """
        将指定列中的缺失值（N/A）填补为0。

        参数:
        df (pd.DataFrame): 包含数据的DataFrame。
        column_names (list): 要填补缺失值的列名列表。

        返回:
        pd.DataFrame: 填补缺失值后的DataFrame。
        """
        # 检查列名是否存在于DataFrame中
        for column_name in column_names:
            if column_name not in df.columns:
                raise ValueError(f"列名 {column_name} 不存在于DataFrame中。")

        # 将指定列中的缺失值填补为0
        df[column_names] = df[column_names].fillna(0)

        return df

    def split_column_to_integer_and_fraction(self, df, column_name):
        """
        将包含括号内小数部分的列拆分为两个新列，分别是整数部分和括号里的小数部分。

        参数:
        dataframe (pd.DataFrame): 包含数据的DataFrame。
        column_name (str): 要拆分的列名。

        返回:
        pd.DataFrame: 拆分后的DataFrame。
        """
        # 检查列名是否存在于DataFrame中
        if column_name not in df.columns:
            raise ValueError(f"列名 {column_name} 不存在于DataFrame中。")

        # 提取整数部分和括号内的小数部分
        df[column_name + '_num'] = df[column_name].apply(
            lambda x: int(x.split('(')[0]) if isinstance(x, str) else None)
        df[column_name + '_rate'] = df[column_name].apply(
            lambda x: float(x.split('(')[1].split(':')[1].rstrip(')')) if isinstance(x, str) else None)

        return df

    def fill_na_with_mode(self, df, column_names):
        """
        将指定列中的非数值型变量的缺失值（N/A）通过找出众数的方式进行填补。

        参数:
        df (pd.DataFrame): 包含数据的DataFrame。
        column_names (list): 要填补缺失值的列名列表。

        返回:
        pd.DataFrame: 填补缺失值后的DataFrame。
        """
        for column_name in column_names:
            if column_name not in df.columns:
                raise ValueError(f"列名 {column_name} 不存在于DataFrame中。")

            # 计算众数
            mode_value = df[column_name].mode()[0]

            # 将缺失值填补为众数
            df[column_name].fillna(mode_value, inplace=True)

        return df

    def apply_log_transformation(self, df, column_names):
        """
        对指定列进行对数处理（取自然对数）。

        参数:
        df (pd.DataFrame): 包含数据的DataFrame。
        column_names (list): 要进行对数处理的列名列表。

        返回:
        pd.DataFrame: 对数处理后的DataFrame。
        """
        for column_name in column_names:
            if column_name not in df.columns:
                raise ValueError(f"列名 {column_name} 不存在于DataFrame中。")

            # 确保列中没有负值或零值，因为对数处理不允许这些值
            if (df[column_name] <= 0).any():
                return 0

            # 应用对数处理
            df[column_name] = np.log(df[column_name])

        return df

    def use_func(self, df, columns):
        if self.process_name == 'na20':
            return self.fill_na_with_zero(df, columns)
        elif self.process_name == 'na2mean':
            return self.fill_na_with_mean(df, columns)
        elif self.process_name == '%2dot':
            return self.convert_percent_to_decimal(df, columns)
        elif self.process_name == 'na2value':
            return self.fill_na_with_specific_string(df, columns)
        elif self.process_name == 'standervalue':
            return self.split_column_to_integer_and_fraction(df, columns)
        elif self.process_name == 'notnum2zs':
            return self.fill_na_with_mode(df, columns)
        elif self.process_name == 'stdlog':
            return self.apply_log_transformation(df, columns)


class RandomForestTrainer:
    def __init__(self):
        self.train_losses = []  # 记录训练损失
        self.test_losses = []  # 记录测试损失

    def train_save_random_forest(self, df, target_column, model_filename):
        """
        使用随机森林回归模型对数据框中的一列进行预测，并处理包含 NaN 的数据。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名
        model_filename (str): 模型文件名

        返回:
        tuple: 包含三个元素 (predictions, adjusted_r2, fig)
            - predictions: 预测值
            - adjusted_r2: 调整后的 R² 值
            - fig: 可视化图表对象
        """
        # 将目标列与其他列分离
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X = imputer.fit_transform(X)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 初始化随机森林回归模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 训练模型并记录损失
        for epoch in range(100):  # 假设训练10个epoch
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_loss = np.mean((train_predictions - y_train) ** 2)  # MSE
            test_loss = np.mean((test_predictions - y_test) ** 2)  # MSE

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

        # 进行最终预测
        predictions = model.predict(X_test)

        # 计算 R² 值
        r2 = r2_score(y_test, predictions)

        # 计算调整后的 R² 值
        n = X_test.shape[0]  # 样本数量
        p = X_test.shape[1]  # 特征数量
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 保存模型
        joblib.dump(model, model_filename)

        # 可视化训练过程中的预测值与实际值
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions, ax=ax, label='Predicted vs Actual')
        sns.lineplot(x=y_test, y=y_test, color='red', ax=ax, label='Perfect Fit')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()

        # 显示 R² 值和调整后的 R² 值
        ax.text(0.05, 0.95, f'R²: {r2:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        ax.text(0.05, 0.90, f'Adjusted R²: {adjusted_r2:.4f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        # 可视化训练和测试损失
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(self.train_losses, label='Training Loss')
        ax2.plot(self.test_losses, label='Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Squared Error (MSE)')
        ax2.set_title('Training and Test Loss Over Epochs')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        return predictions, adjusted_r2, fig


class GBDTTrainer:
    def __init__(self):
        self.train_losses = []  # 记录训练损失
        self.test_losses = []  # 记录测试损失

    def train_save_gbdt(self, df, target_column, model_filename):
        """
        使用梯度提升决策树回归模型对数据框中的一列进行预测，并处理包含 NaN 的数据。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名
        model_filename (str): 模型文件名

        返回:
        tuple: 包含三个元素 (predictions, adjusted_r2, fig)
            - predictions: 预测值
            - adjusted_r2: 调整后的 R² 值
            - fig: 可视化图表对象
        """
        # 将目标列与其他列分离
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X = imputer.fit_transform(X)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 初始化梯度提升决策树回归模型
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

        # 训练模型并记录损失
        for epoch in range(100):  # 假设训练100个epoch
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_loss = np.mean((train_predictions - y_train) ** 2)  # MSE
            test_loss = np.mean((test_predictions - y_test) ** 2)  # MSE

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

        # 进行最终预测
        predictions = model.predict(X_test)

        # 计算 R² 值
        r2 = r2_score(y_test, predictions)

        # 计算调整后的 R² 值
        n = X_test.shape[0]  # 样本数量
        p = X_test.shape[1]  # 特征数量
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 保存模型
        joblib.dump(model, model_filename)

        # 可视化训练过程中的预测值与实际值
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions, ax=ax, label='Predicted vs Actual')
        sns.lineplot(x=y_test, y=y_test, color='red', ax=ax, label='Perfect Fit')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()

        # 显示 R² 值和调整后的 R² 值
        ax.text(0.05, 0.95, f'R²: {r2:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        ax.text(0.05, 0.90, f'Adjusted R²: {adjusted_r2:.4f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        # 可视化训练和测试损失
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(self.train_losses, label='Training Loss')
        ax2.plot(self.test_losses, label='Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Squared Error (MSE)')
        ax2.set_title('Training and Test Loss Over Epochs')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        return predictions, adjusted_r2, fig

class XGBoostTrainer:
    def __init__(self):
        self.train_losses = []  # 记录训练损失
        self.test_losses = []  # 记录测试损失

    def train_save_xgb(self, df, target_column, model_filename):
        """
        使用 XGBoost 回归模型对数据框中的一列进行预测，并处理包含 NaN 的数据。

        参数:
        df (pd.DataFrame): 数据框对象
        target_column (str): 目标列名
        model_filename (str): 模型文件名

        返回:
        tuple: 包含三个元素 (predictions, adjusted_r2, fig)
            - predictions: 预测值
            - adjusted_r2: 调整后的 R² 值
            - fig: 可视化图表对象
        """
        # 将目标列与其他列分离
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy='mean')  # 你可以选择不同的填充策略，如 'mean', 'median', 'most_frequent'
        X = imputer.fit_transform(X)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 初始化 XGBoost 回归模型
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

        # 训练模型并记录损失
        for epoch in range(10):  # 假设训练100个epoch
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_loss = np.mean((train_predictions - y_train) ** 2)  # MSE
            test_loss = np.mean((test_predictions - y_test) ** 2)  # MSE

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

        # 进行最终预测
        predictions = model.predict(X_test)

        # 计算 R² 值
        r2 = r2_score(y_test, predictions)

        # 计算调整后的 R² 值
        n = X_test.shape[0]  # 样本数量
        p = X_test.shape[1]  # 特征数量
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 保存模型
        joblib.dump(model, model_filename)

        # 可视化训练过程中的预测值与实际值
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions, ax=ax, label='Predicted vs Actual')
        sns.lineplot(x=y_test, y=y_test, color='red', ax=ax, label='Perfect Fit')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()

        # 显示 R² 值和调整后的 R² 值
        ax.text(0.05, 0.95, f'R²: {r2:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        ax.text(0.05, 0.90, f'Adjusted R²: {adjusted_r2:.4f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        # 可视化训练和测试损失
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(self.train_losses, label='Training Loss')
        ax2.plot(self.test_losses, label='Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Squared Error (MSE)')
        ax2.set_title('Training and Test Loss Over Epochs')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        return predictions, adjusted_r2, fig