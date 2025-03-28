import os

def multiply_values(line):
    """将一行中的数值乘以5，并返回修改后的行，保留3位小数"""
    parts = line.split(',')
    # 修改数值部分(忽略第一列)
    for i in range(1, len(parts)):  # 跳过第一个元素，因为它应该是'average'，我们不会处理它
        # 乘以5后保留三位小数
        parts[i] = "{:.3f}".format(float(parts[i]))
    return ','.join(parts[1:])  # 返回时排除'average'这一项

def main():
    input_folder = '/home/fengxiaoyao/acorn_data/words/exp_result/gamma=30/result_dist/average_dist'  # 输入文件夹路径
    output_file = '/home/fengxiaoyao/acorn_data/words/exp_result/gamma=30/result_dist/merged_results_gamma=30_dist.txt'  # 输出文件路径
    
    data_lines = []  # 存储所有处理过的行
    
    # 遍历文件夹中的文件
    for filename in sorted(os.listdir(input_folder)):
        if filename.startswith('average_dist') and filename.endswith('.txt'):
            efs_value = filename.split('_')[1].replace('.txt', '')  # 提取efs值
            
            with open(os.path.join(input_folder, filename), 'r') as infile:
                next(infile)  # 跳过第一行（标题）
                for line in infile:
                    new_line = multiply_values(line.strip())
                    # 在每行的开始添加提取出来的efs值
                    data_lines.append(f"{efs_value},{new_line}")

    # 按照efs值排序，注意提取纯数字部分进行比较
    data_lines.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split(',')[0]))))
    
    with open(output_file, 'w') as outfile:
        # 写入表头
        outfile.write('efs,QPS_HNSW,Recall_HNSW,QPS_ACORN,Recall_ACORN\n')
        # 写入处理后的数据
        for line in data_lines:
            outfile.write(f"{line}\n")

if __name__ == '__main__':
    main()