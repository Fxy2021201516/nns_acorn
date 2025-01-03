import random
import json

# 设置生成的向量数目
num_vectors = 100

vectors = [[random.randint(1, 30) for _ in range(5)] for _ in range(num_vectors)]

# 将每个向量的属性排序
sorted_vectors = [sorted(vector) for vector in vectors]

# 将结果输出为 JSON 格式
with open('query_required_filters_sift1m_nc=12_alpha=0.json', 'w') as f:
    json.dump(sorted_vectors, f, indent=4)
