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
#include <cmath> // for std::mean and std::stdev
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <numeric> // for std::accumulate
#include <set>
#include <sstream> // for ostringstream
#include <thread>
#include "utils.cpp"

// create indices for debugging, write indices to file, and get recall stats for
// all queries
int main(int argc, char* argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout
            << "====================\nSTART: running TEST_ACORN for hnsw, sift data --"
            << nthreads << "cores\n"
            << std::endl;
    // printf("===================\nSTART: running MAKE_INDICES for hnsw
    // --...\n");
    double t0 = elapsed();

    int efc = 40;   // default is 40
    int efs = 1000; //  default is 16
    int k = 10;     // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten
                    // by the dimension of the dataset
    int M;          // HSNW param M TODO change M back
    int M_beta;     // param for compression
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int gamma;
    int n_centroids;
    // int filter = 0;
    std::string dataset; // must be sift1B or sift1M or tripclick
    int test_partitions = 0;
    int step = 10; // 2

    std::string assignment_type = "rand";
    int alpha = 0;
    bool generate_json = false;

    srand(0); // seed for random number generator
    int num_trials = 60;

    size_t N = 0; // N will be how many we truncate nb from sift1M to

    std::vector<std::vector<std::unordered_set<int>>>
            e_coverage; // 第一层查询的次数;第二层：对于每个查询，记录该查询的属性;第三层：对于每个属性，记录哪些向量覆盖了该属性

    int opt;
    { // parse arguments

        if (argc < 6 || argc > 8) {
            fprintf(stderr,
                    "Syntax: %s <number vecs> <gamma> [<assignment_type>] [<alpha>] <dataset> <M> <M_beta>\n",
                    argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);

        gamma = atoi(argv[2]);
        printf("gamma: %d\n", gamma);

        dataset = argv[3];
        printf("dataset: %s\n", dataset.c_str());
        if (dataset != "sift1M" && dataset != "sift1M_test" &&
            dataset != "sift1B" && dataset != "tripclick" &&
            dataset != "paper" && dataset != "paper_rand2m") {
            printf("got dataset: %s\n", dataset.c_str());
            fprintf(stderr,
                    "Invalid <dataset>; must be a value in [sift1M, sift1B]\n");
            exit(1);
        }

        M = atoi(argv[4]);
        printf("M: %d\n", M);

        M_beta = atoi(argv[5]);
        printf("M_beta: %d\n", M_beta);
    }

    // load metadata
    n_centroids = gamma;
    // std::vector<int> metadata =
    //         load_ab(dataset,
    //                 gamma,
    //                 assignment_type,
    //                 N); // 读取属性JSON文件，metadata是属性的vector
    // metadata.resize(N);
    // assert(N == metadata.size());
    // printf("[%.3f s] Loaded metadata, %ld attr's found\n",
    //        elapsed() - t0,
    //        metadata.size());
    printf("enter load_ab_muti\n");
    std::vector<std::vector<int>> metadata_multi =
            load_ab_muti(dataset, n_centroids, assignment_type, N);
    std::cout << "metadata_multi.size():" << metadata_multi.size() << std::endl;
    std::cout << "N:" << N << std::endl;
    metadata_multi.resize(N);
    assert(N == metadata_multi.size());
    printf("[%.3f s] Loaded multi metadata, %ld attr's found\n",
           elapsed() - t0,
           metadata_multi.size());

    size_t nq;
    float* xq;
    // std::vector<int> aq;
    std::vector<std::vector<int>> aq_multi;  // 必须有的属性
    std::vector<std::vector<int>> oaq_multi; // 可选的属性
    { // load query vectors and attributes
        printf("[%.3f s] Loading query vectors and attributes\n",
               elapsed() - t0);

        size_t d2;
        // xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        bool is_base = 0;
        // load_data(dataset, is_base, &d2, &nq, xq);
        std::string filename = get_file_name(dataset, is_base);
        // xq = fvecs_read(filename.c_str(), &d2, &nq);
        //  xq = fvecs_read_one_vector(filename.c_str(), &d2, 100);
        nq = 100;
        xq = fvecs_read_first_n_vectors(filename.c_str(), &d2, nq);
        assert(d == d2 ||
               !"query does not have same dimension as expected 128");
        if (d != d2) {
            d = d2;
        }
        // nq = 100;
        std::cout << "query vecs data loaded, with dim: " << d2 << ", nb=" << nq
                  << std::endl;
        printf("[%.3f s] Loaded query vectors from %s\n",
               elapsed() - t0,
               filename.c_str());
        // aq = load_aq(dataset, n_centroids, alpha, N);
        printf("enter load_aq_multi\n");
        aq_multi = load_aq_multi(dataset, n_centroids, alpha, N);
        oaq_multi = load_oaq_multi(dataset, n_centroids, alpha, N);

        // 初始化 coverage 数组，用于记录每个查询对每个属性的覆盖情况
        e_coverage.resize(oaq_multi.size() + 1); // 为每个查询创建记录
        for (int i = 0; i <= oaq_multi.size(); ++i) {
            // 为每个查询创建属性的记录
            e_coverage[i].resize(105);
        }
        printf("[%.3f s] Loaded %ld %s queries\n",
               elapsed() - t0,
               nq,
               dataset.c_str());
    }
    // nq = 1;
    // int gt_size = 100;
    // if (dataset == "sift1M_test" || dataset == "paper") {
    //     gt_size = 10;
    // }
    // std::vector<faiss::idx_t> gt(gt_size * nq);
    // { // load ground truth
    //     gt = load_gt(dataset, gamma, alpha, assignment_type, N);
    //     printf("[%.3f s] Loaded ground truth, gt_size: %d\n",
    //            elapsed() - t0,
    //            gt_size);
    // }

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
           elapsed() - t0,
           d,
           M,
           N,
           gamma);
    // base HNSW index
    faiss::IndexHNSWFlat base_index(d, M, 1); // gamma = 1
    base_index.hnsw.efConstruction = efc;     // default is 40  in HNSW.capp
    base_index.hnsw.efSearch = efs;           // default is 16 in HNSW.capp

    // ACORN-gamma
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata_multi, M_beta);
    hybrid_index.acorn.efSearch = efs; // default is 16 HybridHNSW.capp
    debug("ACORN index created%s\n", "");

    // ACORN-1
    faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, metadata_multi, M * 2);
    hybrid_index_gamma1.acorn.efSearch = efs; // default is 16 HybridHNSW.capp

    { // populating the database
        std::cout << "====================Vectors====================\n"
                  << std::endl;
        // printf("====================Vectors====================\n");

        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        bool is_base = 1;
        std::string filename = get_file_name(dataset, is_base);
        float* xb = fvecs_read(filename.c_str(), &d2, &nb);
        assert(d == d2 || !"dataset does not dim 128 as expected");
        printf("[%.3f s] Loaded base vectors from file: %s\n",
               elapsed() - t0,
               filename.c_str());

        std::cout << "data loaded, with dim: " << d2 << ", nb=" << nb
                  << std::endl;

        printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
               elapsed() - t0,
               N,
               d2,
               nb);

        // index->add(nb, xb);

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        base_index.add(N, xb);
        printf("[%.3f s] Vectors added to base index \n", elapsed() - t0);
        std::cout << "Base index vectors added: " << nb << std::endl;

        hybrid_index.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);
        std::cout << "Hybrid index vectors added" << nb << std::endl;
        // printf("SKIPPED creating ACORN-gamma\n");

        hybrid_index_gamma1.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index with gamma=1 \n",
               elapsed() - t0);
        std::cout << "Hybrid index with gamma=1 vectors added" << nb
                  << std::endl;

        delete[] xb;
    }

    // write hybrid index and partition indices to files
    {
        std::cout << "====================Write Index====================\n"
                  << std::endl;
        // write hybrid index
        // std::string filename = "hybrid_index" + dataset + ".index";
        std::stringstream filepath_stream;
        if (dataset == "sift1M" || dataset == "sift1B") {
            filepath_stream << "./tmp_multi/hybrid_" << (int)(N / 1000 / 1000)
                            << "m_nc=" << n_centroids
                            << "_assignment=" << assignment_type
                            << "_alpha=" << alpha << ".json";

        } else {
            filepath_stream << "./tmp_multi/" << dataset << "/hybrid"
                            << "_M=" << M << "_efc" << efc << "_Mb=" << M_beta
                            << "_gamma=" << gamma << ".json";
        }
        std::string filepath = filepath_stream.str();
        write_index(&hybrid_index, filepath.c_str());
        printf("[%.3f s] Wrote hybrid index to file: %s\n",
               elapsed() - t0,
               filepath.c_str());

        // write hybrid_gamma1 index
        std::stringstream filepath_stream2;
        if (dataset == "sift1M" || dataset == "sift1B") {
            filepath_stream2 << "./tmp_multi/hybrid_gamma1_"
                             << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                             << "_assignment=" << assignment_type
                             << "_alpha=" << alpha << ".json";

        } else {
            filepath_stream2 << "./tmp_multi/" << dataset << "/hybrid"
                             << "_M=" << M << "_efc" << efc << "_Mb=" << M_beta
                             << "_gamma=" << 1 << ".json";
        }
        std::string filepath2 = filepath_stream2.str();
        write_index(&hybrid_index_gamma1, filepath2.c_str());
        printf("[%.3f s] Wrote hybrid_gamma1 index to file: %s\n",
               elapsed() - t0,
               filepath2.c_str());

        { // write base index
            std::stringstream filepath_stream;
            if (dataset == "sift1M" || dataset == "sift1B") {
                filepath_stream << "./tmp_multi/base_" << (int)(N / 1000 / 1000)
                                << "m_nc=" << n_centroids
                                << "_assignment=" << assignment_type
                                << "_alpha=" << alpha << ".json";

            } else {
                filepath_stream << "./tmp_multi/" << dataset << "/base"
                                << "_M=" << M << "_efc=" << efc << ".json";
            }
            std::string filepath = filepath_stream.str();
            write_index(&base_index, filepath.c_str());
            printf("[%.3f s] Wrote base index to file: %s\n",
                   elapsed() - t0,
                   filepath.c_str());
        }
    }

    { // print out stats
        printf("====================================\n");
        printf("============ BASE INDEX =============\n");
        printf("====================================\n");
        base_index.printStats(false);
        printf("====================================\n");
        printf("============ ACORN INDEX =============\n");
        printf("====================================\n");
        hybrid_index.printStats(false);
    }

    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");
    // double t1 = elapsed();
    printf("==============================================\n");
    printf("====================Search====================\n");
    printf("==============================================\n");
    double t1 = elapsed();

    { // searching the base database
        printf("====================HNSW INDEX====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index, efsearch %d\n",
               elapsed() - t0,
               k,
               nq,
               base_index.hnsw.efSearch);

        // 打印metadata_multi
        // for (int i = 0; i < metadata_multi.size(); i++) {
        //     printf("metadata_multi[%d]: ", i);
        //     for (int val : metadata_multi[i]) {
        //         printf("%d ", val);
        //     }
        //     printf("\n");
        // }
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        std::cout << "here1" << std::endl;
        std::cout << "nn and dis size: " << nns.size() << " " << dis.size()
                  << std::endl;

        double t1 = elapsed();
        base_index.search(nq, xq, k, dis.data(), nns.data());
        double t2 = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        // take max of 5 and nq
        // int nq_print = std::min(5, (int)nq);
        // for (int i = 0; i < nq_print; i++) {
        //     printf("query %2d nn's: ", i);
        //     for (int j = 0; j < k; j++) {
        //         // printf("%7ld (%d) ", nns[j + i * k], metadata.size());
        //         printf("%7ld (%d) ",
        //                nns[j + i * k],
        //                metadata_multi[nns[j + i * k]]);
        //     }
        //     printf("\n     dis: \t");
        //     for (int j = 0; j < k; j++) {
        //         printf("%7g ", dis[j + i * k]);
        //     }
        //     printf("\n");
        //     // exit(0);
        // }
        int nq_print = std::min(100, (int)nq);
        for (int i = 0; i < nq_print; i++) {
            printf("my query %2d nn's: ", i);
            for (int j = 0; j < k; j++) {
                int row_idx = nns[j + i * k]; // 获取行索引
                if (row_idx >= 0 && row_idx < metadata_multi.size()) {
                    // 打印该行中的所有元素
                    printf("%7ld (", nns[j + i * k]);
                    for (int val : metadata_multi[row_idx]) {
                        printf("%d ", val);
                    }
                    printf(") ");
                } else {
                    printf("Invalid row index ");
                }
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis[j + i * k]);
            }
            printf("\n");
        }

        printf("[%.3f s] *** Query time: %f\n", elapsed() - t0, t2 - t1);

        // print number of distance computations
        // printf("[%.3f s] *** Number of distance computations: %ld\n",
        //    elapsed() - t0, base_index.ntotal * nq);
        std::cout << "finished base index examples" << std::endl;
    }

    { // look at stats
        // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
        const faiss::HNSWStats& stats = faiss::hnsw_stats;

        std::cout
                << "============= BASE HNSW QUERY PROFILING STATS ============="
                << std::endl;
        printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);
        std::cout << "n1: " << stats.n1 << std::endl;
        std::cout << "n2: " << stats.n2 << std::endl;
        std::cout << "n3 (number distance comps at level 0): " << stats.n3
                  << std::endl;
        std::cout << "ndis: " << stats.ndis << std::endl;
        std::cout << "nreorder: " << stats.nreorder << std::endl;
        std::cout << "average distance computations per query: "
                  << (float)stats.n3 / stats.n1 << std::endl;
    }

    { // searching the hybrid database
        printf("==================== ACORN INDEX ====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index, efsearch %d\n",
               elapsed() - t0,
               k,
               nq,
               hybrid_index.acorn.efSearch);

        std::vector<faiss::idx_t> nns2(k * nq);
        std::vector<float> dis2(k * nq);
        std::vector<float> cost2(k * nq);

        int query_id = 0; // 记录查询的次数

        // 计算距离并生成 JSON 文件
        float* all_distances = new float[nq * N]; // 存储距离结果
        if (generate_json) {
            hybrid_index.calculate_distances_multi(
                    nq, xq, k, all_distances, nns2.data(), query_id);
            save_distances_to_json(nq, N, all_distances, "distances");
        } else {
            all_distances = read_all_distances("my_dis_of_every_query", nq, N);
        }

        // 计算覆盖率并生成 JSON 文件
        std::vector<std::vector<float>> optional_coverage;
        if (generate_json) {
            calculate_attribute_coverage(
                    metadata_multi, oaq_multi, optional_coverage);
            save_coverage_to_json(
                    optional_coverage, metadata_multi, "optional_coverage");
        } else {
            optional_coverage =
                    read_optional_coverage("my_opattr_coverage", nq, N);
        }

        std::vector<std::vector<float>> all_cost;
        if (generate_json) {
            float alpha = 0.5f; // 设置 alpha 值
            calculate_and_save_cost(
                    "my_dis_of_every_query",
                    "my_opattr_coverage",
                    "my_cost",
                    all_cost,
                    alpha);
        } else {
            all_cost = read_all_cost("my_cost", nq, N);
        }

        double t1_x = elapsed();
        // const SearchParameters* params_index = nullptr;
        // for (int query_id = 0; query_id < nq; query_id++) {
        hybrid_index.search_multi(
                nq,
                xq,
                k,
                cost2.data(),
                nns2.data(),
                aq_multi,
                oaq_multi,
                e_coverage,
                all_cost,
                query_id);
        //}
        double t2_x = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        int nq_print = std::min(100, (int)nq);
        for (int i = 0; i < nq_print; i++) {
            printf("my query %2d nn's: ", i);
            for (int j = 0; j < k; j++) {
                int row_idx = nns2[j + i * k]; // 获取行索引
                if (row_idx >= 0 && row_idx < metadata_multi.size()) {
                    // 打印该行中的所有元素
                    printf("%7ld (", nns2[j + i * k]);
                    for (int val : metadata_multi[row_idx]) {
                        printf("%d ", val);
                    }
                    printf(") ");
                } else {
                    printf("Invalid row index ");
                }
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", cost2[j + i * k]);
            }
            printf("\n");
        }

        printf("[%.3f s] *** Query time: %f\n", elapsed() - t0, t2_x - t1_x);

        std::vector<std::vector<float>> sort_filter_all_cost;
        extract_and_sort_costs(
                all_cost, aq_multi, metadata_multi, sort_filter_all_cost);
        // 计算recall
        if (generate_json) {
            create_directory("my_cost_sort_filter");
            saveAllCostToJSON(sort_filter_all_cost, "my_cost_sort_filter");
        }
        double recall = calculateRecall(sort_filter_all_cost, cost2, nq, k);
        std::cout << "Recall: " << recall << std::endl;

        // 输出整体覆盖率
        // float e_coverage_val = calculate_e_coverage(0, e_coverage);
        // printf("e_coverage: %f\n", e_coverage_val);

        std::cout << "finished hybrid index examples" << std::endl;
    }

    // check here

    { // look at stats
        // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
        const faiss::ACORNStats& stats = faiss::acorn_stats;

        std::cout << "============= ACORN QUERY PROFILING STATS ============="
                  << std::endl;
        printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);
        std::cout << "n1: " << stats.n1 << std::endl;
        std::cout << "n2: " << stats.n2 << std::endl;
        std::cout << "n3 (number distance comps at level 0): " << stats.n3
                  << std::endl;
        std::cout << "ndis: " << stats.ndis << std::endl;
        std::cout << "nreorder: " << stats.nreorder << std::endl;
        // printf("average distance computations per query: %f\n",
        //        (float)stats.n3 / stats.n1);
    }

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}