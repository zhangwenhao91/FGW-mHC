#include <ATen/ops/empty.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

using Shape = std::vector<int64_t>;

static pybind11::cpp_function bind_allocate(const at::Tensor& like,
                                            const Shape& partial_shape,
                                            const Shape& squares_shape,
                                            int64_t batch, int64_t hidden) {
    const auto bf16 = like.options().dtype(at::kBFloat16);
    const auto fp32 = like.options().dtype(at::kFloat);
    return pybind11::cpp_function(
        [=]() {
            return std::make_tuple(
                at::empty(partial_shape, fp32),
                at::empty(squares_shape, fp32),
                at::empty({batch, 24}, fp32),
                at::empty({batch, 4, hidden}, bf16),
                at::empty({batch, hidden}, bf16),
                at::empty({batch, 4}, fp32),
                at::empty({batch, 4, hidden}, bf16));
        }, pybind11::call_guard<pybind11::gil_scoped_release>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bind_allocate", &bind_allocate);
}
