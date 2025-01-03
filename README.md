# FAISS HNSW and ACORN Implementation

## HNSW
### Overview

This document provides an overview of the HNSW (Hierarchical Navigable Small World) and ACORN (Approximate COntent-based Retrieval Network) implementations in the FAISS library. These algorithms are used for efficient similarity search in high-dimensional spaces.

### Code Structure

#### `faiss/impl/HNSW.h`

The `HNSW` struct in [`faiss/impl/HNSW.h`](faiss/impl/HNSW.h) contains the core implementation of the HNSW algorithm. Key methods include:

- `add_with_locks`: Adds a point to all levels up to `pt_level` and builds the link structure.
- `hybrid_add_with_locks`: Similar to `add_with_locks` but for hybrid indices.
- `search`: Searches for the `k` nearest neighbors of a query point.
- `search_level_0`: Searches only in level 0 from a given vertex.
- `hybrid_search`: Searches for the `k` nearest neighbors with additional filtering options.
- `reset`, `clear_neighbor_tables`, `print_neighbor_stats`, `print_edges`: Utility functions for managing and debugging the HNSW structure.

#### `faiss/IndexHNSW.h`

The [`faiss/IndexHNSW.h`](faiss/IndexHNSW.h) file defines several index structures that use HNSW for efficient access:

- `IndexHNSW`: Base class for HNSW indices.
- `IndexHNSWFlat`: Flat index with HNSW structure.
- `IndexHNSWPQ`: PQ (Product Quantization) index with HNSW structure.
- `IndexHNSWSQ`: SQ (Scalar Quantization) index with HNSW structure.
- `IndexHNSW2Level`: Two-level index with HNSW structure.
- `IndexHNSWHybridOld`: Hybrid index inheriting from `IndexHNSWFlat`.

## ACORN

### Overview

ACORN (Approximate COntent-based Retrieval Network) is an indexing method implemented in FAISS for efficient similarity search. It is designed to handle large-scale datasets with high-dimensional vectors. ACORN uses a combination of hashing and graph-based search to quickly approximate nearest neighbors. For more information about ACORN, refer to the paper : [**ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data**](https://dl.acm.org/doi/10.1145/3654923)

### Code Structure

#### `faiss/impl/ACORN.h`

The `ACORN` struct in [`faiss/impl/ACORN.h`](faiss/impl/ACORN.h) contains the core implementation of the ACORN algorithm. Key methods include:

- `add`: Adds a new vector to the index.
- `search`: Searches for the `k` nearest neighbors of a query vector.
- `remove`: Removes a vector from the index.
- `update`: Updates an existing vector in the index.
- `rebuild`: Rebuilds the index to optimize search performance.

#### `faiss/IndexACORN.h`

The [`faiss/IndexACORN.h`](faiss/IndexACORN.h) file defines the index structure that uses ACORN for efficient access:

- `IndexACORN`: Base class for ACORN indices.


### How ACORN Works

1. **Initialization**: The ACORN index is initialized with parameters such as the number of hash tables and the dimensionality of the vectors.
2. **Adding Vectors**: Vectors are added to the index using the `add` method. Each vector is hashed into multiple hash tables.
3. **Searching**: To find the `k` nearest neighbors of a query vector, the `search` method is used. The query vector is hashed, and candidate vectors are retrieved from the hash tables. A graph-based search is then performed to refine the results.
4. **Updating and Removing Vectors**: Vectors can be updated or removed from the index using the `update` and `remove` methods, respectively.
5. **Rebuilding the Index**: The `rebuild` method can be used to optimize the index for better search performance.

### Example Usage

1) Initialize the index
```
d=128;
M=32; 
gamma=12;
M_beta=32;

// ACORN-gamma
faiss::IndexACORNFlat acorn_gamma(d, M, gamma, M_beta);

// ACORN-1
faiss::IndexACORNFlat acorn_1(d, M, 1, M*2);
```
2) Construct the index
```
size_t nb, d2;
std::string filename = // your fvec file
float* xb = fvecs_read(filename.c_str(), &d2, &nb);
assert(d == d2 || !"dataset dimension is not as expected");
acorn_gamma.add(nb, xb);
```

3) Search the index
```
// ... load nq queries, xb
// ... load attribute filters as array aq

std::vector<faiss::idx_t> nns2(k * nq);
std::vector<float> dis2(k * nq);

// create filter_ids_map to specify the passing entities for each predicate
std::vector<char> filter_ids_map(nq * N);
for (int xq = 0; xq < nq; xq++) {
    for (int xb = 0; xb < N; xb++) {
        filter_ids_map[xq * N + xb] = (bool) (metadata[xb] == aq[xq]);
    }
}

// perform efficient hybrid search
acorn_gamma.search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data());
```

## ACORN: Multi-Attribute Support with Required Attributes

### Changes Implemented

#### New Data Structures
Several new data structures have been introduced to manage attribute-related information:
```cpp
std::vector<std::vector<int>> metadata_multi; // Attribute information for all vectors
std::vector<std::vector<int>> aq_multi;       // Required attributes
std::vector<std::vector<int>> oaq_multi;      // Optional attributes
```

#### Modified Functions
A number of functions have been modified or added based on the original ACORN source code. These changes are primarily reflected in functions with a `_multi` suffix. You can find the modifications or additions by searching for `multi` or `fxy_add` within the following files:

- **`test_acorn.cpp`**: This is the test file that includes entry points for both index construction and search functionality.
- **`ACORN.h/ACORN.cpp`**: Contains the core data structures and functions related to ACORN.
- **`IndexACORN.h/IndexACORN.cpp`**: Defines functions that invoke those in `ACORN.h/ACORN.cpp`.
- **`utils.cpp`**: Utility functions for tasks such as file reading, writing, and calculating coverage.

### Running Instructions

1. First, download the code into the ACORN folder. Create a new folder named `acorn_data` at the same level as the repository code. Place the following datasets in this folder:
   - Datasets (can be downloaded via `down_sift1M.py`).
   - `testing_data_multi` (the attributes corresponding to vectors, generated using `generate_attr_multi.py`).
   - `googletest-release-1.12.1.tar.gz` and `json-3.10.4.tar.gz`.
   - `tmp_multi` folder
   
2. Modify the paths in `CMakeLists.txt` and `tests/CMakeLists.txt` to reflect your local setup.
   
3. Adjust the paths in `run_simple_test.sh` as necessary.
   
4. The first time we run the tests, we will need to generate a set of JSON files prefixed with `my_cost`, `my_cost_sort_filter`, etc. This can be done by setting `generate_json = true` in `test_acorn.cpp`.
   
5. Once the setup is complete, we can run the tests using the following command:
```bash
./run_simple_test.sh > output.log 2>&1
```
