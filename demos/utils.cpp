#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <sys/time.h>

#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// added these
#include <arpa/inet.h>
#include <assert.h> /* assert */
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <math.h>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>      // for std::mean and std::stdev
#include <filesystem> // C++17 标准库中的文件系统库
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <numeric> // for std::accumulate
#include <set>
#include <sstream> // for ostringstream
#include <thread>
// #include <format>
// for convenience
using json = nlohmann::json;
/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
        -> wget -r ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
        -> cd ftp.irisa.fr/local/texmex/corpus
        -> tar -xf sift.tar.gz

 * and unzip it to the sudirectory sift1M.
 **/

// MACRO
#define TESTING_DATA_DIR "../acorn_data/testing_data"
#define TESTING_DATA_MULTI_DIR "../acorn_data/testing_data_multi"

#include <zlib.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// using namespace std;

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

float* fvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out) { // 维度、数量
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x; // 返回读取的数据，一个包含 n 行，每行 d 个浮点数的数组
}

// fxy_add
float* fvecs_read_one_vector(
        const char* fname,
        size_t* d_out,
        size_t index) { // 新增 index 参数，用于指定要读取的查询向量的索引
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, 1, sizeof(int), f); // 读取维度信息
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");

    fseek(f, 0, SEEK_SET);

    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");

    size_t n = sz / ((d + 1) * 4); // 计算数据点数量

    *d_out = d;

    // 判断索引是否越界
    if (index >= n) {
        fprintf(stderr,
                "Index %zu is out of range. There are only %zu vectors in the file.\n",
                index,
                n);
        abort();
    }

    // 移动到所需的查询向量位置
    fseek(f,
          sizeof(int) + index * (d + 1) * sizeof(float),
          SEEK_SET); // 跳过文件头并定位到索引位置

    // 读取一个向量
    float* query_vector = new float[d];
    size_t nr = fread(query_vector, sizeof(float), d, f);
    assert(nr == d || !"could not read the vector");

    fclose(f);

    return query_vector; // 返回单个向量
}

// fxy_add
float* fvecs_read_first_n_vectors(
        const char* fname,
        size_t* d_out,
        size_t n) { // 读取前 n 个向量
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, 1, sizeof(int), f); // 读取维度信息
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");

    size_t total_vectors = sz / ((d + 1) * 4); // 计算文件中数据点数量

    *d_out = d;

    // 如果 n 超过文件中的向量数量，修正为最大数量
    if (n > total_vectors) {
        n = total_vectors;
    }

    // 申请存储 n 个向量的内存
    float* vectors = new float[n * (d + 1)];

    // 读取前 n 个向量
    fseek(f, sizeof(int), SEEK_SET); // 跳过维度头
    size_t nr = fread(
            vectors, sizeof(float), n * (d + 1), f); // 读取前 n 个向量及其头
    assert(nr == n * (d + 1) || !"could not read the vectors");

    // 去掉每个向量的头部数据
    for (size_t i = 0; i < n; i++) {
        memmove(vectors + i * d,
                vectors + 1 + i * (d + 1),
                d * sizeof(*vectors));
    }

    fclose(f);

    return vectors; // 返回读取的前 n 个向量
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

// get file name to load data vectors from
std::string get_file_name(std::string dataset, bool is_base) {
    if (dataset == "sift1M" || dataset == "sift1M_test") {
        return std::string("../acorn_data/Datasets/sift1M/sift_") +
                (is_base ? "base" : "query") + ".fvecs";
    } else if (dataset == "sift1B") {
        return std::string("../acorn_data/Datasets/sift1B/bigann_") +
                (is_base ? "base_10m" : "query") + ".fvecs";
    } else if (dataset == "tripclick") {
        return std::string("../acorn_data/Datasets/tripclick/") +
                (is_base ? "base_vecs_tripclick"
                         : "query_vecs_tripclick_min100") +
                ".fvecs";
    } else if (dataset == "paper" || dataset == "paper_rand2m") {
        return std::string("../acorn_data/Datasets/paper/") +
                (is_base ? "paper_base" : "paper_query") + ".fvecs";
    } else {
        std::cerr << "Invalid datset in get_file_name" << std::endl;
        return "";
    }
}

// return name is in arg file_path
void get_index_name(
        int N,
        int n_centroids,
        std::string assignment_type,
        float alpha,
        int M_beta,
        std::string& file_path) {
    std::stringstream filepath_stream;
    filepath_stream << "./tmp/hybrid_" << (int)(N / 1000 / 1000)
                    << "m_nc=" << n_centroids
                    << "_assignment=" << assignment_type << "_alpha=" << alpha
                    << "Mb=" << M_beta << ".json";
    // copy filepath_stream to file_path
    file_path = filepath_stream.str();
}

/*******************************************************
 * Added for debugging
 *******************************************************/
const int debugFlag = 1;

void debugTime() {
    if (debugFlag) {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        struct tm* tm_info = localtime(&tval.tv_sec);
        char timeBuff[25] = "";
        strftime(timeBuff, 25, "%H:%M:%S", tm_info);
        char timeBuffWithMilli[50] = "";
        sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
        std::string timestamp(timeBuffWithMilli);
        std::cout << timestamp << std::flush;
    }
}

// needs atleast 2 args always
//   alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__);
#define debug(fmt, ...)                             \
    do {                                            \
        if (debugFlag == 1) {                       \
            fprintf(stdout, "--" fmt, __VA_ARGS__); \
        }                                           \
        if (debugFlag == 2) {                       \
            debugTime();                            \
            fprintf(stdout,                         \
                    "%s:%d:%s(): " fmt,             \
                    __FILE__,                       \
                    __LINE__,                       \
                    __func__,                       \
                    __VA_ARGS__);                   \
        }                                           \
    } while (0)

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*******************************************************
 * performance testing helpers
 *******************************************************/
std::pair<float, float> get_mean_and_std(std::vector<float>& times) {
    // compute mean
    float total = 0;
    // for (int num: times) {
    for (int i = 0; i < times.size(); i++) {
        // printf("%f, ", times[i]); // for debugging
        total = total + times[i];
    }
    float mean = (total / times.size());

    // compute stdev from variance, using computed mean
    float result = 0;
    for (int i = 0; i < times.size(); i++) {
        result = result + (times[i] - mean) * (times[i] - mean);
    }
    float variance = result / (times.size() - 1);
    // for debugging
    // printf("variance: %f\n", variance);

    float std = std::sqrt(variance);

    // return
    return std::make_pair(mean, std);
}

// ground truth labels @gt, results to evaluate @I with @nq queries, returns
// @gt_size-Recall@k where gt had max gt_size NN's per query
float compute_recall(
        std::vector<faiss::idx_t>& gt,
        int gt_size,
        std::vector<faiss::idx_t>& I,
        int nq,
        int k,
        int gamma = 1) {
    // printf("compute_recall params: gt.size(): %ld, gt_size: %d, I.size():
    // %ld, nq: %d, k: %d, gamma: %d\n", gt.size(), gt_size, I.size(), nq, k,
    // gamma);

    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (int i = 0; i < nq; i++) { // loop over all queries
        // int gt_nn = gt[i * k];
        std::vector<faiss::idx_t>::const_iterator first =
                gt.begin() + i * gt_size;
        std::vector<faiss::idx_t>::const_iterator last =
                gt.begin() + i * gt_size + (k / gamma);
        std::vector<faiss::idx_t> gt_nns_tmp(first, last);
        // if (gt_nns_tmp.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns_tmp.size());
        // }

        // gt_nns_tmp.resize(k); // truncate if gt_size > k
        std::set<faiss::idx_t> gt_nns(gt_nns_tmp.begin(), gt_nns_tmp.end());
        // if (gt_nns.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns.size());
        // }

        for (int j = 0; j < k; j++) { // iterate over returned nn results
            if (gt_nns.count(I[i * k + j]) != 0) {
                // if (I[i * k + j] == gt_nn) {
                if (j < 1 * gamma)
                    n_1++;
                if (j < 10 * gamma)
                    n_10++;
                if (j < 100 * gamma)
                    n_100++;
            }
        }
    }
    // BASE ACCURACY
    // printf("* Base HNSW accuracy relative to exact search:\n");
    // printf("\tR@1 = %.4f\n", n_1 / float(nq) );
    // printf("\tR@10 = %.4f\n", n_10 / float(nq));
    // printf("\tR@100 = %.4f\n", n_100 / float(nq)); // not sure why this is
    // always same as R@10 printf("\t---Results for %ld queries, k=%d, N=%ld,
    // gt_size=%d\n", nq, k, N, gt_size);
    return (n_10 / float(nq));
}

template <typename T>
void log_values(std::string annotation, std::vector<T>& values) {
    std::cout << annotation;
    for (int i = 0; i < values.size(); i++) {
        std::cout << values[i];
        if (i < values.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FOR CORRELATION TESTING
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<T> load_json_to_vector(std::string filepath) {
    // Open the JSON file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file" << std::endl;
        // return 1;
    }

    // Parse the JSON data
    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON data from " << filepath << ": "
                  << e.what() << std::endl;
        // return 1;
    }

    // Convert data to a vector
    std::vector<T> v = data.get<std::vector<T>>();

    // print size
    std::cout << "metadata or vector loaded from json, size: " << v.size()
              << std::endl;
    return v;
}

// fxy_add
template <typename T>
std::vector<std::vector<T>> load_json_to_vector_multi(std::string filepath) {
    // Open the JSON file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file" << std::endl;
        // return 1;
    }

    // Parse the JSON data
    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON data from " << filepath << ": "
                  << e.what() << std::endl;
        // throw std::runtime_error("Failed to parse JSON data");
    }

    std::vector<std::vector<T>> v;
    for (const auto& array : data) {
        std::vector<T> sub_vector = array.get<std::vector<T>>();
        v.push_back(sub_vector);
    }

    // Print the size of the loaded data
    std::cout << "Loaded vector of vectors from JSON, size: " << v.size()
              << std::endl;
    return v;
}

std::vector<int> load_aq(
        std::string dataset,
        int n_centroids,
        int alpha,
        int N) {
    if (dataset == "sift1M" || dataset == "sift1B") {
        assert((alpha == -2 || alpha == 0 || alpha == 2) ||
               !"alpha must be value in [-2, 0, 2]");

        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/query_filters_sift"
                        << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                        << "_alpha=" << alpha << ".json";
        std::string filepath = filepath_stream.str();

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR
                        << "/query_filters_tripclick_sample_subset_min100.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "sift1M_test") {
        // return a vector of all int 5 with lenght N
        std::vector<int> v(N, 5);
        printf("made query filters with value %d, length %ld\n",
               v[0],
               v.size());
        return v;

    } else if (dataset == "paper") {
        std::vector<int> v(N, 5);
        printf("made query filters with value %d, length %ld\n",
               v[0],
               v.size());
        return v;
    } else if (dataset == "paper_rand2m") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR
                        << "/query_filters_paper_rand2m_nc=12_alpha=0.json";
        std::string filepath = filepath_stream.str();

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());
        return v;

    } else {
        std::cerr << "Invalid dataset in load_aq" << std::endl;
        return std::vector<int>();
    }
}

// fxy_add
std::vector<std::vector<int>> load_aq_multi(
        std::string dataset,
        int n_centroids,
        int alpha,
        int N) {
    if (dataset == "sift1M" || dataset == "sift1B") {
        assert((alpha == -2 || alpha == 0 || alpha == 2) ||
               !"alpha must be value in [-2, 0, 2]");

        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR
                        << "/query_required_filters_sift"
                        << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                        << "_alpha=" << alpha << ".json";
        std::string filepath = filepath_stream.str();

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR
                        << "/query_filters_tripclick_sample_subset_min100.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else {
        std::cerr << "Invalid dataset in load_aq" << std::endl;
        return std::vector<std::vector<int>>();
    }
}

// fxy_add
std::vector<std::vector<int>> load_oaq_multi(
        std::string dataset,
        int n_centroids,
        int alpha,
        int N) {
    if (dataset == "sift1M" || dataset == "sift1B") {
        assert((alpha == -2 || alpha == 0 || alpha == 2) ||
               !"alpha must be value in [-2, 0, 2]");

        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR
                        << "/query_optional_filters_sift"
                        << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                        << "_alpha=" << alpha << ".json";
        std::string filepath = filepath_stream.str();

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded query optional attributes from: %s\n", filepath.c_str());

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR
                        << "/query_filters_tripclick_sample_subset_min100.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded query optional attributes from: %s\n", filepath.c_str());
        return v;
    } else {
        std::cerr << "Invalid dataset in load_aq" << std::endl;
        return std::vector<std::vector<int>>();
    }
}

// assignment_type can be "rand", "soft", "soft_squared", "hard"
std::vector<int> load_ab(
        std::string dataset,
        int n_centroids,
        std::string assignment_type,
        int N) {
    // Compose File Name
    if (dataset == "sift1M" || dataset == "sift1B") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/base_attrs_sift"
                        << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                        << "_assignment=" << assignment_type << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "sift1M_test") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/sift_attr" << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());
        return v;

    } else if (dataset == "paper") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/paper_attr.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        return v;

    } else if (dataset == "paper_rand2m") {
        std::stringstream filepath_stream;
        filepath_stream
                << TESTING_DATA_DIR
                << "/base_attrs_paper_rand2m_nc=12_assignment=rand.json";
        std::string filepath = filepath_stream.str();

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/base_attrs_tripclick.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else {
        std::cerr << "Invalid dataset in load_ab" << std::endl;
        return std::vector<int>();
    }
}

// fxy_add 载入多属性JSON文件
std::vector<std::vector<int>> load_ab_muti(
        std::string dataset,
        int n_centroids,
        std::string assignment_type,
        int N) {
    // Compose File Name
    if (dataset == "sift1M" || dataset == "sift1B") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR << "/base_attrs_sift"
                        << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                        << "_assignment=" << assignment_type << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "sift1M_test") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR << "/sift_attr" << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());
        return v;

    } else if (dataset == "paper") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR << "/paper_attr.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        return v;

    } else if (dataset == "paper_rand2m") {
        std::stringstream filepath_stream;
        filepath_stream
                << TESTING_DATA_MULTI_DIR
                << "/base_attrs_paper_rand2m_nc=12_assignment=rand.json";
        std::string filepath = filepath_stream.str();

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_MULTI_DIR
                        << "/base_attrs_tripclick.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<std::vector<int>> v =
                load_json_to_vector_multi<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else {
        std::cerr << "Invalid dataset in load_ab" << std::endl;
        return std::vector<std::vector<int>>();
    }
}

// assignment_type can be "rand", "soft", "soft_squared", "hard"
// alpha can be -2, 0, 2
std::vector<faiss::idx_t> load_gt(
        std::string dataset,
        int n_centroids,
        int alpha,
        std::string assignment_type,
        int N) {
    if (dataset == "sift1M" || dataset == "sift1B") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/gt_sift"
                        << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                        << "_assignment=" << assignment_type
                        << "_alpha=" << alpha << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "sift1M_test") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/sift_gt_5.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;

    } else if (dataset == "paper") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/paper_gt_5.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;

    } else if (dataset == "paper_rand2m") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream
                << TESTING_DATA_DIR
                << "/gt_paper_rand2m_nc=12_assignment=rand_alpha=0.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR
                        << "/gt_tripclick_sample_subset_min100.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else {
        std::cerr << "Invalid dataset in load_gt" << std::endl;
        return std::vector<faiss::idx_t>();
    }
}

// fxy_add:计算当前查询的整体覆盖率
float calculate_e_coverage(
        int query_idx,
        const std::vector<std::vector<std::unordered_set<int>>>& e_coverage) {
    const auto& query_coverage = e_coverage[query_idx]; // 第二层，查询的属性集

    // 统计当前查询覆盖的属性数
    int covered_attributes = 0;
    // int total_attributes = query_coverage.size(); // 当前查询的属性总数
    int total_attributes = 30;
    for (const auto& attribute_coverage : query_coverage) {
        if (!attribute_coverage.empty()) {
            covered_attributes++; // 如果属性被覆盖，计数加1
        }
    }

    // 计算覆盖率
    if (total_attributes == 0) {
        return 0.0f; // 防止除以0的情况
    }
    return static_cast<float>(covered_attributes) / total_attributes;
}

// fxy_add
void create_directory(const std::string& dir_name) {
    if (mkdir(dir_name.c_str(), 0777) == -1) {
        std::cerr << "Failed to create directory: " << dir_name << std::endl;
    } else {
        std::cout << "Directory created: " << dir_name << std::endl;
    }
}

// fxy_add
void save_single_query_to_json(
        size_t query_idx,       // 当前查询的索引
        size_t ntotal,          // 存储向量的数量
        const float* distances, // 距离数组
        const std::string& filename_prefix) {
    // 创建一个 JSON 对象来存储单个查询的距离
    nlohmann::json j;

    // 提取查询向量与所有存储向量的距离
    std::vector<float> query_distances;
    for (size_t j = 0; j < ntotal; ++j) {
        query_distances.push_back(distances[query_idx * ntotal + j]);
    }

    // 将查询的距离存储到 JSON 数组中
    j = query_distances;

    // 确保文件夹 'dis_of_every_query' 存在，如果不存在则创建
    create_directory("../acorn_data/my_dis_of_every_query");

    // 生成查询的文件名，格式为
    // "dis_of_every_query/query_<query_idx>.json"
    std::string filename = "../acorn_data/my_dis_of_every_query/" +
            filename_prefix + "_query_" + std::to_string(query_idx) + ".json";

    // 打开文件并写入 JSON 数据
    std::ofstream output_file(filename);
    if (output_file.is_open()) {
        output_file << j.dump(4); // 写入文件，格式化缩进为 4
        output_file.close();
        std::cout << "Distances for query " << query_idx << " saved to "
                  << filename << std::endl;
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

// fxy_add
void save_distances_to_json(
        size_t nq,              // 查询数量
        size_t ntotal,          // 存储向量数量
        const float* distances, // 存储距离的数组
        const std::string& filename_prefix) {
    for (size_t i = 0; i < nq; ++i) {
        save_single_query_to_json(i, ntotal, distances, filename_prefix);
    }
};

// fxy_add
float* read_all_distances(const std::string& folder_path, size_t nq, size_t N) {
    // 创建一个存储所有距离的数组
    float* all_distances = new float[nq * N];
    size_t index = 0;

    // 遍历所有的文件
    for (size_t i = 0; i < nq; i++) {
        // 构造文件路径
        std::string file_path =
                folder_path + "/distances_query_" + std::to_string(i) + ".json";

        // 打开文件
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Could not open file: " << file_path << std::endl;
            continue;
        }

        // 解析 JSON 文件
        json j;
        file >> j; // 读取文件内容到 json 对象

        // 检查文件格式是否正确
        if (j.is_array() && j.size() == N) {
            for (size_t j_index = 0; j_index < N; j_index++) {
                all_distances[index++] =
                        j[j_index].get<float>(); // 读取浮动数值
            }
        } else {
            std::cerr << "File " << file_path
                      << " format is incorrect or data size does not match N!"
                      << std::endl;
        }

        file.close();
    }

    return all_distances;
}

// fxy_add 排序距离
void sort_distances(
        float* all_distances, // 所有距离的数组
        size_t nq,            // 查询数量
        size_t N) {           // 存储向量数量
    // 遍历每个查询
    for (size_t i = 0; i < nq; i++) {
        // 使用 std::sort 对每个查询的距离进行排序
        std::sort(all_distances + i * N, all_distances + (i + 1) * N);
    }
}

// fxy_add 保存排序后的距离
void save_sorted_distances(
        float* all_distances, // 所有距离的数组
        size_t nq,            // 查询数量
        size_t N,             // 存储向量数量
        const std::string& filename_prefix) {
    // 遍历每个查询
    for (size_t i = 0; i < nq; i++) {
        // 创建一个 JSON 对象来存储排序后的距离
        nlohmann::json j;

        // 提取当前查询的排序后的距离
        std::vector<float> query_distances;
        for (size_t j = 0; j < N; ++j) {
            query_distances.push_back(all_distances[i * N + j]);
        }

        // 将排序后的距离存储到 JSON 数组中
        j = query_distances;

        // 确保文件夹 'sorted_dis_of_every_query' 存在，如果不存在则创建
        create_directory("../acorn_data/my_sorted_dis_of_every_query");

        // 生成查询的文件名，格式为
        // "sorted_dis_of_every_query/query_<query_idx>.json"
        std::string filename = "../acorn_data/my_sorted_dis_of_every_query/" +
                filename_prefix + "_query_" + std::to_string(i) + ".json";

        // 打开文件并写入 JSON 数据
        std::ofstream output_file(filename);
        if (output_file.is_open()) {
            output_file << j.dump(4); // 写入文件，格式化缩进为 4
            output_file.close();
            std::cout << "Sorted distances for query " << i << " saved to "
                      << filename << std::endl;
        } else {
            std::cerr << "Failed to open file: " << filename << std::endl;
        }
    }
}

// fxy_add 计算每个向量在每次查询下的属性覆盖率
void calculate_attribute_coverage(
        const std::vector<std::vector<int>>& metadata_multi, // 向量的属性
        const std::vector<std::vector<int>>& oaq_multi,      // 查询的属性
        std::vector<std::vector<float>>& coverage) {
    std::cout << "enter calculate_attribute_coverage" << std::endl;

    size_t query_count = oaq_multi.size();
    size_t vector_count = metadata_multi.size();

    coverage.clear(); // Clear coverage at the beginning

    // 预处理 oaq_multi 中的属性集，避免重复计算
    std::vector<std::unordered_set<int>> oaq_sets(query_count);
    for (size_t q = 0; q < query_count; ++q) {
        for (int attr : oaq_multi[q]) {
            oaq_sets[q].insert(attr);
        }
    }
    std::cout << "finish preprocessing oaq_multi" << std::endl;

    for (size_t q = 0; q < query_count; ++q) {
        std::cout << "q: " << q << std::endl;
        const auto& oaq_set = oaq_sets[q]; // 当前查询尽量包含的属性

        std::vector<float> query_coverage; // 存储当前查询下所有向量的覆盖率

        // 遍历每个向量的属性
        for (size_t i = 0; i < vector_count; ++i) {
            const auto& metadata = metadata_multi[i]; // 当前向量的属性
            float covered_count = 0;

            // 使用 unordered_set 来存储 metadata 中的属性，避免重复
            std::unordered_set<int> metadata_set(
                    metadata.begin(), metadata.end());

            // 计算当前向量属性与查询包含属性的交集
            for (int attr : oaq_set) {
                if (metadata_set.find(attr) != metadata_set.end()) {
                    covered_count += 1;
                }
            }

            // 计算覆盖率：覆盖的属性数 / 查询的属性数
            float coverage_rate =
                    covered_count / static_cast<float>(oaq_set.size());
            query_coverage.push_back(coverage_rate);
        }

        // 将当前查询下的所有向量的覆盖率加入到最终结果
        coverage.push_back(query_coverage);
    }
}

// fxy_add 将覆盖率结果保存到 JSON 文件
void save_coverage_to_json(
        const std::vector<std::vector<float>>& coverage,     // 属性覆盖率
        const std::vector<std::vector<int>>& metadata_multi, // 向量的属性
        const std::string& base_filename) { // 输出文件名的基础部分
    create_directory("../acorn_data/my_opattr_coverage");

    // 遍历每个查询的覆盖率
    for (size_t q = 0; q < coverage.size(); ++q) {
        // 为每个查询生成一个独立的 JSON 对象
        nlohmann::json j;

        // 遍历每个向量的属性和覆盖率
        for (size_t i = 0; i < coverage[q].size(); ++i) {
            // 保留两位小数，使用 std::round
            float formatted_coverage =
                    std::round(coverage[q][i] * 100.0f) / 100.0f;

            // 将浮动数值转换为带两位小数的字符串
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(2) << formatted_coverage;
            j.push_back(stream.str()); // 将格式化后的字符串存入 JSON 数据
        }

        // 生成查询的文件名，并确保路径指向文件夹 'optional_attr_coverage'
        std::ostringstream filename_stream;
        filename_stream << "my_opattr_coverage/" << base_filename << "_query_"
                        << q << ".json";
        std::string filename = filename_stream.str();

        // 打开文件并写入 JSON 数据
        std::ofstream output_file(filename);
        if (output_file.is_open()) {
            // 写入格式化后的 JSON 数据
            output_file << j.dump(4); // 使用 4 个空格缩进格式化输出
            output_file.close();
            std::cout << "Coverage data for query " << q << " saved to "
                      << filename << std::endl;
        } else {
            std::cerr << "Failed to open file: " << filename << std::endl;
        }
    }
}

// fxy_add 读取覆盖率数据
std::vector<std::vector<float>> read_optional_coverage(
        const std::string& folder_path,
        size_t nq,
        size_t N) {
    // 创建一个二维 vector 来存储所有的距离值
    std::vector<std::vector<float>> optional_coverage;

    // 遍历所有文件
    for (size_t i = 0; i < nq; i++) {
        // 构造文件路径
        std::string file_path = folder_path + "/optional_coverage_query_" +
                std::to_string(i) + ".json";

        // 打开文件
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Could not open file: " << file_path << std::endl;
            continue;
        }

        // 解析 JSON 文件
        json j;
        file >> j; // 读取文件内容到 json 对象

        // 检查文件格式是否正确
        if (j.is_array() && j.size() == N) {
            std::vector<float> coverage_row;
            for (size_t j_index = 0; j_index < N; j_index++) {
                // 将字符串转换为 float 并存入 coverage_row
                coverage_row.push_back(
                        std::stof(j[j_index].get<std::string>()));
            }
            // 将当前行数据加入到 optional_coverage 中
            optional_coverage.push_back(coverage_row);
        } else {
            std::cerr << "File " << file_path
                      << " format is incorrect or data size does not match N!"
                      << std::endl;
        }

        file.close();
    }

    return optional_coverage;
}

// fxy_add 计算成本并保存到 JSON 文件
void calculate_and_save_cost(
        const std::string& distance_folder,        // 存储距离数据的文件夹路径
        const std::string& coverage_folder,        // 存储覆盖率数据的文件夹路径
        const std::string& output_folder,          // 输出文件夹路径
        std::vector<std::vector<float>>& all_cost, // 存储所有的 cost
        float alpha) {                             // alpha 值

    // 创建输出文件夹 'my_cost' 如果不存在
    create_directory(output_folder);
    // 存储所有的cost

    // 遍历文件夹中的查询文件
    for (size_t q = 0;; ++q) {
        // 生成距离和覆盖率文件名
        std::ostringstream dist_filename_stream;
        dist_filename_stream << distance_folder << "/distances_query_" << q
                             << ".json";
        std::string dist_filename = dist_filename_stream.str();

        std::ostringstream coverage_filename_stream;
        coverage_filename_stream << coverage_folder
                                 << "/optional_coverage_query_" << q << ".json";
        std::string coverage_filename = coverage_filename_stream.str();

        // 打开距离文件
        std::ifstream dist_file(dist_filename);
        if (!dist_file.is_open()) {
            break; // 如果文件未找到，说明所有文件已读取完毕，跳出循环
        }

        // 打开覆盖率文件
        std::ifstream coverage_file(coverage_filename);
        if (!coverage_file.is_open()) {
            std::cerr << "Failed to open coverage file: " << coverage_filename
                      << std::endl;
            continue;
        }

        // 读取 JSON 数据
        nlohmann::json dist_json;
        dist_file >> dist_json;
        nlohmann::json coverage_json;
        coverage_file >> coverage_json;

        // 计算结果并保存
        std::vector<float> cost_values;
        float max_dist = *std::max_element(dist_json.begin(), dist_json.end());
        size_t num_vectors = dist_json.size(); // 假设距离和覆盖率的数量一致
        for (size_t i = 0; i < num_vectors; ++i) {
            // 获取对应的距离 d 和覆盖率 c
            float d = dist_json[i];

            // 将 d 归一化到 [0, 1] 范围
            float d_normalized = d / max_dist;

            // float c = coverage_json[i];
            float c = std::stof(coverage_json[i].get<std::string>());

            // 计算 alpha * d - (1 - alpha) * c
            float cost = alpha * d_normalized - (1 - alpha) * c;
            cost_values.push_back(cost);
        }

        // 生成输出文件名
        std::ostringstream output_filename_stream;
        output_filename_stream << output_folder << "/cost_query_" << q
                               << ".json";
        std::string output_filename = output_filename_stream.str();

        // 写入计算结果到 JSON 文件
        std::ofstream output_file(output_filename);
        if (output_file.is_open()) {
            output_file
                    << nlohmann::json(cost_values).dump(4); // 使用 4 个空格缩进
            output_file.close();
            std::cout << "Cost for query " << q << " saved to "
                      << output_filename << std::endl;
        } else {
            std::cerr << "Failed to open output file: " << output_filename
                      << std::endl;
        }
        all_cost.push_back(cost_values);
    }
}

// fxy_add 读取cost数据
std::vector<std::vector<float>> read_all_cost(
        const std::string& folder_path,
        size_t nq,
        size_t N) {
    // 创建一个二维 vector 来存储所有的成本值
    std::vector<std::vector<float>> all_cost;

    // 遍历所有文件
    for (size_t i = 0; i < nq; i++) {
        // 构造文件路径
        std::string file_path =
                folder_path + "/cost_query_" + std::to_string(i) + ".json";

        // 打开文件
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Could not open file: " << file_path << std::endl;
            continue;
        }

        // 解析 JSON 文件
        json j;
        file >> j; // 读取文件内容到 json 对象

        // 检查文件格式是否正确
        if (j.is_array() && j.size() == N) {
            std::vector<float> cost_row;
            for (size_t j_index = 0; j_index < N; j_index++) {
                // 将浮动数值添加到当前行
                cost_row.push_back(j[j_index].get<float>());
            }
            // 将当前行的数据添加到 all_cost 中
            all_cost.push_back(cost_row);
        } else {
            std::cerr << "File " << file_path
                      << " format is incorrect or data size does not match N!"
                      << std::endl;
        }

        file.close();
    }

    return all_cost;
}

// fxy_add
bool binary_search(const std::vector<int>& vec, int target) {
    int left = 0;
    int right = vec.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (vec[mid] == target) {
            return true; // 找到目标元素
        } else if (vec[mid] < target) {
            left = mid + 1; // 在右半部分查找
        } else {
            right = mid - 1; // 在左半部分查找
        }
    }
    return false; // 如果没有找到，返回 false
}

// fxy_add 判断向量是否包含所有必需的属性
bool has_required_attributes(
        const std::vector<int>& vector_attributes,
        const std::vector<int>& required_attributes) {
    for (int attr : required_attributes) {
        if (std::find(
                    vector_attributes.begin(), vector_attributes.end(), attr) ==
            vector_attributes.end()) {
            return false;
        }
    }
    return true;
}

// fxy_add 处理并提取符合要求的向量的dist，并排序后存入
// sort_filter_all_distances
void extract_and_sort_distances(
        const float* all_distances,
        const std::vector<std::vector<int>>& aq_multi,
        const std::vector<std::vector<int>>& metadata_multi,
        int nq, // 查询数量
        int N,  // 向量数量
        std::vector<std::vector<float>>& sort_filter_all_distances) {
    // 遍历每个查询
    std::cout << "enter extract_and_sort_distances" << std::endl;
    int query_count = aq_multi.size();
    std::cout << "query_count: " << query_count << std::endl;
    for (size_t query_index = 0; query_index < query_count; query_index++) {
        // 获取当前查询的必需属性
        const std::vector<int>& required_attributes = aq_multi[query_index];

        // 临时容器，用来存储符合条件的向量的索引和对应的距离
        std::vector<std::pair<int, float>> valid_vectors;

        // 遍历所有向量，检查是否符合当前查询的属性要求
        int vector_count = metadata_multi.size();
        for (size_t vector_index = 0; vector_index < vector_count;
             vector_index++) {
            const std::vector<int>& vector_attributes =
                    metadata_multi[vector_index];

            // 如果当前向量的属性符合要求
            if (has_required_attributes(
                        vector_attributes, required_attributes)) {
                // 计算距离在 all_distances 中的索引
                int distance_index = query_index * N + vector_index;

                // 将符合条件的向量的索引和对应的距离保存到 valid_vectors 中
                valid_vectors.push_back(
                        {static_cast<int>(vector_index),
                         all_distances[distance_index]});
            }
        }

        // 按照距离对符合条件的向量进行排序（升序）
        std::sort(
                valid_vectors.begin(),
                valid_vectors.end(),
                [](const std::pair<int, float>& a,
                   const std::pair<int, float>& b) {
                    return a.second < b.second; // 比较距离，升序排序
                });

        // 创建一个存储排序后距离的数组
        std::vector<float> sorted_distances;
        for (const auto& vec : valid_vectors) {
            sorted_distances.push_back(vec.second); // 只保存距离
        }

        // 将排序后的距离添加到 sort_filter_all_distances 中
        sort_filter_all_distances.push_back(sorted_distances);
    }
}

// fxy_add 处理并提取符合要求的向量的 cost，并排序后存入 sort_filter_all_cost
void extract_and_sort_costs(
        const std::vector<std::vector<float>>& all_cost,
        const std::vector<std::vector<int>>& aq_multi,
        const std::vector<std::vector<int>>& metadata_multi,
        std::vector<std::vector<float>>& sort_filter_all_cost) {
    // 遍历每个查询
    std::cout << "enter extract_and_sort_costs" << std::endl;
    int query_count = aq_multi.size();
    std::cout << "query_count: " << query_count << std::endl;
    for (size_t query_index = 0; query_index < query_count; query_index++) {
        // 获取当前查询的必需属性
        const std::vector<int>& required_attributes = aq_multi[query_index];

        // 临时容器，用来存储符合条件的向量的索引和对应的 cost
        std::vector<std::pair<int, float>> valid_vectors;

        // 遍历所有向量，检查是否符合当前查询的属性要求
        int vector_count = metadata_multi.size();
        for (size_t vector_index = 0; vector_index < vector_count;
             vector_index++) {
            const std::vector<int>& vector_attributes =
                    metadata_multi[vector_index];

            // 如果当前向量的属性符合要求
            if (has_required_attributes(
                        vector_attributes, required_attributes)) {
                // 将符合条件的向量的索引和对应的 cost 保存到 valid_vectors 中
                valid_vectors.push_back(
                        {static_cast<int>(vector_index),
                         all_cost[query_index][vector_index]});
            }
        }

        // 按照 cost 对符合条件的向量进行排序（升序）
        std::sort(
                valid_vectors.begin(),
                valid_vectors.end(),
                [](const std::pair<int, float>& a,
                   const std::pair<int, float>& b) {
                    return a.second < b.second; // 比较 cost，升序排序
                });

        // 创建一个存储排序后向量索引的数组
        std::vector<float> sorted_costs;
        for (const auto& vec : valid_vectors) {
            sorted_costs.push_back(vec.second); // 只保存 cost
        }

        // 将排序后的向量索引添加到 sort_filter_all_cost 中
        sort_filter_all_cost.push_back(sorted_costs);
    }
}

// fxy_add 处理cost：排序后存入 sort_all_cost
void sort_costs(
        const std::vector<std::vector<float>>& all_cost,
        std::vector<std::vector<float>>& sort_all_cost) {
    // 遍历每个查询
    for (const auto& cost : all_cost) {
        // 使用 std::sort 对每个查询的 cost 进行排序
        std::vector<float> sorted_cost = cost;
        std::sort(sorted_cost.begin(), sorted_cost.end());
        sort_all_cost.push_back(sorted_cost);
    }
}

// fxy_add 保存排序后的 cost 到 JSON 文件
void saveAllCostToJSON(
        const std::vector<std::vector<float>>& all_cost,
        const std::string& folder_path) {
    for (size_t i = 0; i < all_cost.size(); ++i) {
        // 将每个查询的 cost 存为 JSON 文件
        std::string file_name =
                folder_path + "/cost_sort_query_" + std::to_string(i) + ".json";
        std::ofstream file(file_name);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << file_name << std::endl;
            continue;
        }

        // 将数据写入 JSON 文件
        json query_json = all_cost[i]; // 自动将 std::vector 转换为 JSON 数组
        file << query_json.dump(4);    // 格式化输出，缩进为 4 空格
        file.close();
    }
}

// fxy_add 计算recall
double calculateRecall(
        const std::vector<std::vector<float>>& all_cost,
        const std::vector<float>& cost2,
        int nq,
        int k,
        float epsilon = 1e-5) {
    int correct = 0;
    int total = 0;

    for (int i = 0; i < nq; ++i) {
        // 提取第 i 个查询的最短距离
        std::vector<float> query_cost(
                cost2.begin() + i * k, cost2.begin() + (i + 1) * k);

        // 提取排序后的 all_cost 的第 i 个查询结果
        const std::vector<float>& sorted_cost = all_cost[i];
        int pre_correct = correct;

        // 检查 query_cost 的每个值是否在 sorted_cost 的前 k 个元素中
        for (const float& cost : query_cost) {
            total++;
            // 使用近似比较而不是精确匹配
            bool found = false;
            for (int j = 0; j < k; ++j) {
                if (std::abs(sorted_cost[j] - cost) < epsilon) {
                    found = true;
                    break;
                }
            }
            if (found) {
                ++correct;
            }
        }

        std::cout << "i: " << i << " correct: " << correct - pre_correct
                  << std::endl;
    }

    return static_cast<double>(correct) / total;
}