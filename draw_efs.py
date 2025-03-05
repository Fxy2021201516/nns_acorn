import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置后端
import matplotlib.pyplot as plt

# 读取数据
data = []
with open('/home/fengxiaoyao/acorn_data_sift1m/efs_results/all_efs.txt', 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  # 跳过标题行
        parts = line.strip().split(',')
        efs = int(parts[0])
        qps_hnsw = float(parts[1])
        recall_hnsw = float(parts[2].replace('(', '').replace(')', ''))
        qps_acorn = float(parts[3])
        recall_acorn = float(parts[4].replace('(', '').replace(')', ''))
        data.append((efs, qps_hnsw, recall_hnsw, qps_acorn, recall_acorn))

# 分离数据以便绘图
efs_values, qps_hnsw_values, recall_hnsw_values, qps_acorn_values, recall_acorn_values = zip(*data)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(qps_hnsw_values, recall_hnsw_values, label='HNSW', marker='o')
plt.plot(qps_acorn_values, recall_acorn_values, label='ACORN', marker='x')

plt.title('QPS vs Recall Comparison')
plt.xlabel('QPS (Queries Per Second)')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('output.png')  # 将图表保存为PNG文件