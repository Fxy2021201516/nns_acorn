import json
import random

# 读取 JSON 文件
with open('/home/fengxiaoyao/acorn_data/sift1m/testing_data_multi/origin_required.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取每个向量的第一个属性作为唯一属性
new_data = [[vec[0]] for vec in data if vec]

# 保存到新的 JSON 文件
with open('new_data.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

# 生成新的属性文件，每个向量对应 3 个不重复的属性
def generate_attributes(existing_values, num_attributes=3, value_range=30):
    return random.sample([x for x in range(1,value_range) if x not in existing_values], num_attributes)

# 读取 new_data.json
with open('new_data.json', 'r', encoding='utf-8') as f:
    existing_data = json.load(f)

# 生成新属性数据
attribute_data = [generate_attributes(set(vec)) for vec in existing_data]

# 保存到新的 JSON 文件
with open('attributes.json', 'w', encoding='utf-8') as f:
    json.dump(attribute_data, f, ensure_ascii=False, indent=4)

print("处理完成，结果已保存到 new_data.json 和 attributes.json")
