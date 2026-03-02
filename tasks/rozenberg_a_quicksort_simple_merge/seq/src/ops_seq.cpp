#include "rozenberg_a_quicksort_simple_merge/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "rozenberg_a_quicksort_simple_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace rozenberg_a_quicksort_simple_merge {

RozenbergAQuicksortSimpleMergeSEQ::RozenbergAQuicksortSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &elem : in) {
    GetInput().push_back(elem);
  }

  GetOutput().clear();
}

bool RozenbergAQuicksortSimpleMergeSEQ::ValidationImpl() {
  return (!(GetInput().empty())) && (GetOutput().empty());
}

bool RozenbergAQuicksortSimpleMergeSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return GetOutput().size() == GetInput().size();
}

int RozenbergAQuicksortSimpleMergeSEQ::Partition(InType &data, int left, int right) {
  int pivot = data[left + (right - left) / 2];
  int i = left - 1;
  int j = right + 1;
  while (true) {
    do {
      i++;
    } while (data[i] < pivot);
    do {
      j--;
    } while (data[j] > pivot);
    if (i >= j) {
      return j;
    }
    std::swap(data[i], data[j]);
  }
}

void RozenbergAQuicksortSimpleMergeSEQ::Quicksort(InType &data, int left, int right) {
  while (left < right) {
    int q = Partition(data, left, right);
    if (q - left < right - q) {
      Quicksort(data, left, q);
      left = q + 1;
    } else {
      Quicksort(data, q + 1, right);
      right = q;
    }
  }
}

bool RozenbergAQuicksortSimpleMergeSEQ::RunImpl() {
  InType data = GetInput();
  Quicksort(data, 0, data.size() - 1);
  GetOutput() = data;
  return true;
}

bool RozenbergAQuicksortSimpleMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_quicksort_simple_merge
