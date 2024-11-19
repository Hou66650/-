import pandas as pd
import re

# 定义一个函数，将百分数字符串转换为浮点数
def percentage_to_float(percentage_str):
    if isinstance(percentage_str, str):
        # 去掉百分号并转换为浮点数
        return float(percentage_str.strip('%')) / 100
    return percentage_str  # 如果不是字符串，则保持原样

# 假设你要处理的列名为 'percentage_column'



# 定义一个函数来替换整数值
def replace_integer_values(value):
    # 使用正则表达式提取数值部分
    match = re.match(r'(\d+)\(', value)
    if match:
        # 检查提取的数值是否为整数
        if match.group(1).isdigit():
            return '100(1:0)'
    return value
# 读取CSV文件
df = pd.read_csv('test_2.csv', encoding='latin1')
# 读取数据
# 检查数据结构
print(df.head())
# 确保数据没有合并到一行
if df.shape[0] == 1 and ',' in df.columns:
    print("数据可能被错误地合并到一行，请检查数据源或导入设置。")
else:
    print("数据结构正常。")
# 将N/A值替换为当前列的上一行数据值
df.fillna(method='ffill', inplace=True)
df.to_csv('test_2m.csv', index=False)
# # 处理 'Greening rate' 列
# # 应用函数到指定的列
# df['Greening rate'] = df['Greening rate'].apply(percentage_to_float)
# # 将 'N/A' 替换为 '100(1:0)'
# # # 将空白值替换为 '100(1:0)'
# # df['parking space'] = df['parking space'].replace(r'^\s*$', '100(1:0)', regex=True
# 应用替换函数到 'Area' 列
df['parking space'] = df['parking space'].apply(replace_integer_values)
# 使用字符串方法提取数值和比值部分
df['parking_num'] = df['parking space'].str.extract(r'(\d+)\(').astype(int)
df['parking_rate'] = df['parking space'].str.extract(r'\((\d+):(\d+)\)').apply(lambda x: int(x[1]) / int(x[0]), axis=1)
df.to_csv('3f.csv', index=False)
# 计算每列的中位数
median_values = df.median()

# 将N/A值替换为所在列的中位数
df_filled = df.fillna(median_values)

# 保存清理后的数据到新的CSV文件
df_filled.to_csv('Appendix 1_clean_p.csv', index=False)