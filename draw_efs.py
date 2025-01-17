import matplotlib.pyplot as plt

# 设置非交互式后端
plt.switch_backend('Agg')  # 使用 Agg 后端

# 读取文件并解析数据
recall_hsnw = []
qps_hsnw = []
recall_acorn = []
qps_acorn = []

with open('all_efs.csv', 'r') as file:
    lines = file.readlines()
    # 跳过第一行（标题行）
    for line in lines[1:]:
        # 按逗号分割每一行
        parts = line.strip().split(',')
        # 提取数据
        recall_hsnw.append(float(parts[2]))  # Recall_HNSW
        qps_hsnw.append(float(parts[1]))     # QPS_HNSW
        recall_acorn.append(float(parts[4])) # Recall_ACORN
        qps_acorn.append(float(parts[3]))    # QPS_ACORN

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制HNSW的折线图
plt.plot(recall_hsnw, qps_hsnw, label='HNSW', marker='o')

# 绘制ACORN的折线图
plt.plot(recall_acorn, qps_acorn, label='ACORN', marker='s')

# 添加标题和标签
plt.title('QPS vs Recall for HNSW and ACORN')
plt.xlabel('Recall')
plt.ylabel('QPS')

# 添加图例
plt.legend()

# 保存图像
plt.savefig('qps_vs_recall.png')
print("图像已保存为 qps_vs_recall.png")