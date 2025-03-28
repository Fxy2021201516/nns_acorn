export debugSearchFlag=0
#! /bin/bash

rm -rf build

cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

make -C build -j faiss
make -C build utils
make -C build test_acorn


now=$(date +"%m-%d-%Y")
N=1000000
gamma=12
dataset=sift1M
M=32 
M_beta=64

parent_dir=../acorn_data/${dataset}/${now}_${dataset}  

rm -rf ../acorn_data/${dataset}/${now}_${dataset}  

mkdir -p ${parent_dir}                      

# 创建一个汇总文件，记录所有 efs 的 QPS 和 Recall
summary_file="${parent_dir}/summary_all_efs.txt"
echo "efs,QPS_HNSW,Recall_HNSW,QPS_ACORN,Recall_ACORN" > ${summary_file}

# 循环测试 efs 值
for efs in $(seq 10 10 1000); do
    dir=${parent_dir}/MB${M_beta}_efs${efs}  # 在目录名中加入 efs 值
    mkdir -p ${dir}                          

    TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt

    # 运行测试，传递 efs 参数
    ./build/demos/test_acorn $N $gamma $dataset $M $M_beta $efs &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt

    TZ='America/Los_Angeles' date +"End time: %H:%M" &>> ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt

    # 从日志文件中提取 QPS 和 Recall
    qps_acorn=$(grep "QPS:" ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt | grep "ACORN" | awk -F'QPS:' '{print $2}' | awk '{print $1}')
    recall_acorn=$(grep "Recall:" ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt | grep "ACORN" | awk '{print $NF}')
    qps_hnsw=$(grep "QPS:" ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt | grep "HNSW" | awk -F'QPS:' '{print $2}' | awk '{print $1}')
    recall_hnsw=$(grep "Recall:" ${dir}/summary_sift_n=${N}_gamma=${gamma}_efs=${efs}.txt | grep "HNSW" | awk '{print $NF}')

    # 将结果追加到汇总文件中
    echo "${efs},${qps_hnsw},${recall_hnsw},${qps_acorn},${recall_acorn}" >> ${summary_file}
done


# #!/bin/bash
# export debugSearchFlag=0

# # 编译优化参数
# rm -rf build
# cmake -DFAISS_ENABLE_GPU=OFF \
#       -DFAISS_ENABLE_PYTHON=OFF \
#       -DBUILD_TESTING=ON \
#       -DBUILD_SHARED_LIBS=ON \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_CXX_FLAGS="-march=native -O3" \
#       -B build
# make -C build -j faiss
# make -C build utils
# make -C build test_acorn

# # 运行参数
# now=$(date +"%m-%d-%Y")
# N=1000000
# gamma=30
# dataset=sift1M
# M=32 
# M_beta=64

# # 使用内存文件系统加速I/O
# parent_dir="/dev/shm/${dataset}/${now}_${dataset}"
# mkdir -p "${parent_dir}"

# # 预创建目录
# efs_values=($(seq 10 10 1000))
# for efs in "${efs_values[@]}"; do
#     mkdir -p "${parent_dir}/MB${M_beta}_efs${efs}"
# done

# # 定义任务函数
# run_task() {
#     local task_id=$1  # 任务编号
#     local efs=$2      # efs 值
#     local dir="${parent_dir}/MB${M_beta}_efs${efs}"
#     local log_file="${dir}/summary_efs${efs}.txt"
    
#     # 打印任务进度（实时输出到标准错误）
#     echo "[Progress] Task ${task_id}/${total_tasks}: efs=${efs} (Start at $(date '+%Y-%m-%d %H:%M:%S'))" >&2
    
#     # 绑定到特定CPU核心
#     taskset -c $((task_id % 36)) ./build/demos/test_acorn \
#         ${N} ${gamma} ${dataset} ${M} ${M_beta} ${efs} > "${log_file}" 2>&1
    
#     # 打印任务完成信息（实时输出到标准错误）
#     echo "[Progress] Task ${task_id}/${total_tasks}: efs=${efs} (End at $(date '+%Y-%m-%d %H:%M:%S'))" >&2
# }

# # 导出函数以便 parallel 调用
# export -f run_task
# export parent_dir N gamma dataset M M_beta

# # 分批次执行（每组12任务，共约8组）
# max_parallel=36
# group_size=12
# total_tasks=${#efs_values[@]}  # 总任务数

# # 使用 parallel 调用任务函数，并显示进度条
# printf "%s\n" "${efs_values[@]}" | parallel --ungroup --bar -j $group_size "
#     run_task {#} {}
# "

# # 结果持久化
# persistent_dir="../acorn_data/${dataset}/${now}_${dataset}"
# mkdir -p "${persistent_dir}"
# rsync -a --remove-source-files "${parent_dir}/" "${persistent_dir}/"