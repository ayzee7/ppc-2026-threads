#include "vasiliev_m_shell_sort_batcher_merge/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "vasiliev_m_shell_sort_batcher_merge/common/include/common.hpp"

namespace vasiliev_m_shell_sort_batcher_merge {

VasilievMShellSortBatcherMergeOMP::VasilievMShellSortBatcherMergeOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMShellSortBatcherMergeOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMShellSortBatcherMergeOMP::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VasilievMShellSortBatcherMergeOMP::RunImpl() {
  auto &vec = GetInput();
  const size_t n = vec.size();

  if (vec.empty()) {
    return false;
  }

  std::vector<size_t> bounds = ChunkBoundaries(n, omp_get_max_threads());
  size_t chunk_count = bounds.size() - 1;

  ShellSort(vec, bounds);

  std::vector<ValType> buffer(n);
  for (size_t size = 1; size < chunk_count; size *= 2) {
    BatcherMerge(vec, buffer, bounds, size);
    vec.swap(buffer);
  }

  GetOutput() = vec;
  return true;
}

bool VasilievMShellSortBatcherMergeOMP::PostProcessingImpl() {
  return true;
}

std::vector<size_t> VasilievMShellSortBatcherMergeOMP::ChunkBoundaries(size_t vec_size, int chunks_) {
  size_t chunks = std::max(1, std::min(chunks_, static_cast<int>(vec_size)));

  std::vector<size_t> bounds;
  bounds.reserve(chunks + 1);

  size_t chunk_size = vec_size / chunks;
  size_t remainder = vec_size % chunks;

  bounds.push_back(0);

  for (size_t i = 0; i < chunks; i++) {
    if (i < remainder) {
      bounds.push_back(bounds.back() + chunk_size + 1);
    } else {
      bounds.push_back(bounds.back() + chunk_size);
    }
  }
  return bounds;
}

void VasilievMShellSortBatcherMergeOMP::ShellSort(std::vector<ValType> &vec, std::vector<size_t> &bounds) {
  size_t chunk_count = bounds.size() - 1;

#pragma omp parallel for default(none) shared(vec, bounds, chunk_count) schedule(static)
  for (size_t chunk = 0; chunk < chunk_count; chunk++) {
    size_t first = bounds[chunk];
    size_t last = bounds[chunk + 1];
    size_t n = last - first;

    for (size_t gap = n / 2; gap > 0; gap /= 2) {
      for (size_t i = first + gap; i < last; i++) {
        ValType tmp = vec[i];
        size_t j = i;
        while (j >= first + gap && vec[j - gap] > tmp) {
          vec[j] = vec[j - gap];
          j -= gap;
        }
        vec[j] = tmp;
      }
    }
  }
}

void VasilievMShellSortBatcherMergeOMP::BatcherMerge(std::vector<ValType> &vec, std::vector<ValType> &buffer,
                                                     std::vector<size_t> &bounds, size_t size) {
  size_t chunk_count = bounds.size() - 1;

#pragma omp parallel for default(none) shared(vec, buffer, bounds, size, chunk_count) schedule(static)
  for (size_t l_chunk = 0; l_chunk < chunk_count; l_chunk += (2 * size)) {
    size_t l = l_chunk;
    size_t mid = std::min(l + size, chunk_count);
    size_t r = std::min(l + (2 * size), chunk_count);

    size_t start = bounds[l];
    size_t middle = bounds[mid];
    size_t end = bounds[r];
    auto out = buffer.begin() + start;

    if (mid == r) {
      std::copy(vec.begin() + start, vec.begin() + end, out);
    } else {
      std::merge(vec.begin() + start, vec.begin() + middle, vec.begin() + middle, vec.begin() + end, out);
    }
  }
}

}  // namespace vasiliev_m_shell_sort_batcher_merge
