import os.path
import os

# SIFT1M 数据集的文件下载链接
links = [
    'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz',
]

# 创建用于存放下载和解压文件的目录
os.makedirs('downloads', exist_ok=True)
os.makedirs('sift1m', exist_ok=True)

# 遍历文件链接，下载和解压
for link in links:
    name = link.rsplit('/', 1)[-1]  # 提取文件名
    filename = os.path.join('downloads', name)  # 拼接下载路径
    if not os.path.isfile(filename):  # 如果文件不存在，则下载
        print('Downloading: ' + filename)
        try:
            os.system('wget --output-document=' + filename + ' ' + link)
        except Exception as inst:
            print(inst)
            print('  Encountered unknown error. Continuing.')
    else:
        print('Already downloaded: ' + filename)

    # 解压文件
    if filename.endswith('.tar.gz'):  # 如果是 tar.gz 文件
        command = 'tar -zxf ' + filename + ' --directory sift1m'
    else:  # 如果是 gz 文件
        command = 'cat ' + filename + ' | gzip -dc > sift1m/' + name.replace(".gz", "")
    
    print("Unpacking file:", command)
    os.system(command)
