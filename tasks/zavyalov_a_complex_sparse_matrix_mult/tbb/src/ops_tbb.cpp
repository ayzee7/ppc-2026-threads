#include "zavyalov_a_complex_sparse_matrix_mult/tbb/include/ops_tbb.hpp"
#include "zavyalov_a_complex_sparse_matrix_mult/common/include/common.hpp"

#include <tbb/tbb.h>

#include <atomic>
#include <numeric>
#include <util/include/util.hpp>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

namespace zavyalov_a_compl_sparse_matr_mult  {

ZavyalovAComplSparseMatrMultTBB::ZavyalovAComplSparseMatrMultTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ZavyalovAComplSparseMatrMultTBB::ValidationImpl() {
  const auto &matr_a = std::get<0>(GetInput());
  const auto &matr_b = std::get<1>(GetInput());
  return matr_a.width == matr_b.height;
}

bool ZavyalovAComplSparseMatrMultTBB::PreProcessingImpl() {
  return true;
}

bool ZavyalovAComplSparseMatrMultTBB::RunImpl() {

  //std::atomic<int> counter(0);
  //tbb::parallel_for(0, ppc::util::GetNumThreads(), [&](int /*i*/) { counter++; });

  const auto &matr_a = std::get<0>(GetInput());
  const auto &matr_b = std::get<1>(GetInput());

  GetOutput() = matr_a * matr_b;
  
  return true;
}

bool ZavyalovAComplSparseMatrMultTBB::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_compl_sparse_matr_mult 
