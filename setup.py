from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
has_cuda = torch.cuda.is_available()

# ✅ Include both local and system-wide Eigen path
include_dirs = [
    os.path.join(ROOT, "mast3r_slam/backend/include"),
    os.path.join(ROOT, "thirdparty/eigen"),  # Still here in case some headers exist locally
    "/usr/include/eigen3"                    # ✅ Add system-wide Eigen path
]

# Source files
sources = [
    "mast3r_slam/backend/src/gn.cpp",
]

# Compilation flags
extra_compile_args = {
    "cxx": ["-O3"],
}

# Build extensions depending on CUDA availability
if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources += [
        "mast3r_slam/backend/src/gn_kernels.cu",
        "mast3r_slam/backend/src/matching_kernels.cu",
    ]

    extra_compile_args["nvcc"] = [
        "-O3",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
    ]

    ext_modules = [
        CUDAExtension(
            name="mast3r_slam_backends",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    raise RuntimeError("CUDA not found, cannot compile backend!")

setup(
    name="mast3r_slam_backends",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
