/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

namespace faiss
{

   struct IndexHNSW;

   struct ReconstructFromNeighbors
   {
      typedef HNSW::storage_idx_t storage_idx_t;

      const IndexHNSW &index;
      size_t M;   // number of neighbors
      size_t k;   // number of codebook entries
      size_t nsq; // number of subvectors
      size_t code_size;
      int k_reorder; // nb to reorder. -1 = all

      std::vector<float> codebook; // size nsq * k * (M + 1)

      std::vector<uint8_t> codes; // size ntotal * code_size
      size_t ntotal;
      size_t d, dsub; // derived values

      explicit ReconstructFromNeighbors(
          const IndexHNSW &index,
          size_t k = 256,
          size_t nsq = 1);

      /// codes must be added in the correct order and the IndexHNSW
      /// must be populated and sorted
      void add_codes(size_t n, const float *x);

      size_t compute_distances(
          size_t n,
          const idx_t *shortlist,
          const float *query,
          float *distances) const;

      /// called by add_codes
      void estimate_code(const float *x, storage_idx_t i, uint8_t *code) const;

      /// called by compute_distances
      void reconstruct(storage_idx_t i, float *x, float *tmp) const;

      void reconstruct_n(storage_idx_t n0, storage_idx_t ni, float *x) const;

      /// get the M+1 -by-d table for neighbor coordinates for vector i
      void get_neighbor_table(storage_idx_t i, float *out) const;
   };

   /** The HNSW index is a normal random-access index with a HNSW
    * link structure built on top */

   struct IndexHNSW : Index
   {
      typedef HNSW::storage_idx_t storage_idx_t;

      // the link strcuture
      HNSW hnsw;

      // the sequential storage
      bool own_fields;
      Index *storage;

      ReconstructFromNeighbors *reconstruct_from_neighbors;

      explicit IndexHNSW(
          int d = 0,
          int M = 32,
          int gamma = 1,
          MetricType metric = METRIC_L2);
      explicit IndexHNSW(Index *storage, int M = 32, int gamma = 1);
      explicit IndexHNSW(
          std::vector<std::vector<int>> &metadata_multi,
          int d = 0,
          int M = 32,
          int gamma = 1,
          MetricType metric = METRIC_L2);
      explicit IndexHNSW(
          Index *storage,
          std::vector<std::vector<int>> &metadata_multi,
          int M = 32,
          int gamma = 1);

      ~IndexHNSW() override;

      // add n vectors of dimension d to the index, x is the matrix of vectors
      void add(idx_t n, const float *x) override;

      /// Trains the storage if needed
      void train(idx_t n, const float *x) override;

      /// entry point for search
      void search(
          idx_t n,
          const float *x,
          idx_t k,
          float *distances,
          idx_t *labels,
          const SearchParameters *params = nullptr) const override;
      void search_multi(
          idx_t n,
          const float *x,
          idx_t k,
          float *distances,
          idx_t *labels,
          const std::vector<std::vector<int>> aq_multi,
          std::vector<std::vector<float>> &all_cost,
          int query_id,
          const std::vector<std::vector<int>> &metadata_multi,
          bool dis_or_cost_in_search,
          const SearchParameters *params = nullptr) const;

      // assume equality predicates only
      void partition_search(
          idx_t n,
          const float *x,
          idx_t k,
          std::vector<faiss::IndexHNSW *> &partition_indices,
          int num_partitions,
          float *distances,
          idx_t *labels,
          int *filters,
          const SearchParameters *params = nullptr) const;

      void reconstruct(idx_t key, float *recons) const override;

      void reset() override;

      void shrink_level_0_neighbors(int size);

      /** Perform search only on level 0, given the starting points for
       * each vertex.
       *
       * @param search_type 1:perform one search per nprobe, 2: enqueue
       *                    all entry points
       */
      void search_level_0(
          idx_t n,
          const float *x,
          idx_t k,
          const storage_idx_t *nearest,
          const float *nearest_d,
          float *distances,
          idx_t *labels,
          int nprobe = 1,
          int search_type = 1) const;

      /// alternative graph building
      void init_level_0_from_knngraph(int k, const float *D, const idx_t *I);

      /// alternative graph building
      void init_level_0_from_entry_points(
          int npt,
          const storage_idx_t *points,
          const storage_idx_t *nearests);

      // reorder links from nearest to farthest
      void reorder_links();

      void link_singletons();

      // added for debugging
      void printStats(bool print_edge_list = false, bool is_hybrid = false);
   };

   /** Flat index topped with with a HNSW structure to access elements
    *  more efficiently.
    */

   struct IndexHNSWFlat : IndexHNSW
   {
      IndexHNSWFlat();
      IndexHNSWFlat(int d, int M, int gamma = 1, MetricType metric = METRIC_L2);
      // IndexHNSWFlat(std::vector<std::vector<int>>& metadata,int d, int M, int gamma = 1, MetricType metric = METRIC_L2);
   };

   /** PQ index topped with with a HNSW structure to access elements
    *  more efficiently.
    */
   struct IndexHNSWPQ : IndexHNSW
   {
      IndexHNSWPQ();
      IndexHNSWPQ(int d, int pq_m, int M);
      void train(idx_t n, const float *x) override;
   };

   /** SQ index topped with with a HNSW structure to access elements
    *  more efficiently.
    */
   struct IndexHNSWSQ : IndexHNSW
   {
      IndexHNSWSQ();
      IndexHNSWSQ(
          int d,
          ScalarQuantizer::QuantizerType qtype,
          int M,
          MetricType metric = METRIC_L2);
   };

   /** 2-level code structure with fast random access
    */
   struct IndexHNSW2Level : IndexHNSW
   {
      IndexHNSW2Level();
      IndexHNSW2Level(Index *quantizer, size_t nlist, int m_pq, int M);

      void flip_to_ivf();

      /// entry point for search
      void search(
          idx_t n,
          const float *x,
          idx_t k,
          float *distances,
          idx_t *labels,
          const SearchParameters *params = nullptr) const override;
   };

   /**************************************************************
    * HYBRID INDEX
    **************************************************************/
   /** Inherits from IndexHNSWFlat
    */
   // TODO - change metadata from int* to generic type
   struct IndexHNSWHybridOld : IndexHNSWFlat
   {
      IndexHNSWHybridOld();
      IndexHNSWHybridOld(
          int d,
          int M,
          int gamma,
          std::vector<int> &metadata,
          MetricType metric = METRIC_L2);

      void add(idx_t n, const float *x) override;

      // this doesn't override normal search since definition has a filter param -
      // search is overloaded
      void search(
          idx_t n,
          const float *x,
          idx_t k,
          float *distances,
          idx_t *labels,
          int *filters,
          Operation op = EQUAL,
          std::string regex = "",
          const SearchParameters *params = nullptr) const;
   };

} // namespace faiss
