# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import argparse
import dace
import numpy as np
from pathlib import Path
import subprocess
from textwrap import dedent
import os


N = dace.symbol('N')

def find_access_node_by_name(sdfg, name):
  """ Finds the first data node by the given name"""
  return next((n, s) for n, s in sdfg.all_nodes_recursive()
              if isinstance(n, dace.nodes.AccessNode) and name == n.data)
def find_map_by_name(sdfg, name):
    """ Finds the first map entry node by the given name """
    return next((n, s) for n, s in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and name == n.label)

@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("--simulator", type=str, default='banshee')
    args = parser.parse_args()

    # Initialize arrays
    a = np.random.rand()
    x = np.random.rand(args.N)
    y = np.random.rand(args.N)
    expected = y.copy()
    
    axpy(a, x, expected)

    # Call the program (the value of N is inferred by dace automatically)
    sdfg = axpy.to_sdfg()
    
    # TODO:
    # Load elements of X and Y with SSR streamers
    # find_access_node_by_name(sdfg, 'X')[0].desc(sdfg).storage = dace.dtypes.StorageType.Snitch_SSR
    # find_access_node_by_name(sdfg, 'Y')[0].desc(sdfg).storage = dace.dtypes.StorageType.Snitch_SSR

    # Execute parallel
    #find_map_by_name(sdfg, 'multiplication')[0].schedule = dace.ScheduleType.Snitch_Multicore
    
    from dace.codegen.targets.snitch import SnitchCodeGen
    code, header = SnitchCodeGen.gen_code_snitch(sdfg)
    
    N = args.N
    
    generated_dir = Path('generated_axpy_snitch').resolve()
    
    dace_root = Path(f"{dace.__path__[0]}/..").resolve()
    fallback_snitch_root = dace_root / "../snitch"
    snitch_root = os.environ.get('SNITCH_ROOT', fallback_snitch_root)
    if not Path(snitch_root).is_dir():
        raise RuntimeError(f"SNITCH_ROOT is not found at {snitch_root}")
    
    fallback_snitch_toolchain = dace_root / '../cmake/my-toolchain-llvm.cmake'
    snitch_toolchain = os.environ.get('SNITCH_CMAKE_TOOLCHAIN', fallback_snitch_toolchain)
    if not Path(snitch_toolchain).is_file():
        raise RuntimeError(f"SNITCH_CMAKE_TOOLCHAIN is not found at {snitch_toolchain}")
    
    verilator_path = snitch_root / 'hw/system/snitch_cluster/bin/snitch_cluster.vlt'
    if not Path(verilator_path).is_file():
        raise RuntimeError(f"verilator is not found at {verilator_path}")
    
    banshee_path = snitch_root / 'sw/banshee/target/debug/banshee'
    if not Path(banshee_path).is_file():
        raise RuntimeError(f"banshee is not found at {banshee_path}")
    
    banshee_config = snitch_root / 'sw/banshee/config/snitch_cluster.yaml'
    if not Path(banshee_config).is_file():
        raise RuntimeError(f"banshee config is not found at {banshee_config}")
    
    
    generated_dir.mkdir(exist_ok=True)

    axpy_test = dedent(f"""
        #include "axpy.h"
        #include "snrt.h"
        #include "omp.h"
        #include "dm.h"
        #include "printf.h"
        
        double X[{N}] = {{ {', '.join([str(v) for v in x])} }};
        double Y[{N}] = {{ {', '.join([str(v) for v in y])} }};
        double Z[{N}] = {{ {', '.join([str(v) for v in expected])} }};
        
        int main() {{
            unsigned core_idx = snrt_cluster_core_idx();
            unsigned core_num = snrt_cluster_core_num();
            
            int err = 0;
            
            __snrt_omp_bootstrap(core_idx);
            
            double A = {a};
            int N = {N};

            axpyHandle_t handle = __dace_init_axpy(A, N);

            __program_axpy(handle, X, Y, A, N);
            
            for (int i = 0; i < N; i++) {{
                if (std::abs(Y[i] - Z[i]) >= 1e-5) {{
                    printf("Mismatch: %f %f\\n", Y[i], Z[i]);
                    err = 1;
                    break;
                }}
            }}
            
            __dace_exit_axpy(handle);

            printf("Done, err %d\\n", (int)err);
            
            __snrt_omp_destroy(core_idx);
                
            return err;
        }}
    """)


    # Write code to files
    with open(generated_dir / "axpy.cpp", "w") as fd:
        fd.write(code)
    with open(generated_dir / "axpy.h", "w") as fd:
        fd.write(header)
    with open(generated_dir / "axpy_test.cpp", "w") as fd:
        fd.write(axpy_test)
    with open(generated_dir / "CMakeLists.txt", "w") as fd:
        fd.write(dedent(f"""
            cmake_minimum_required(VERSION 3.13)
            project(Axpy LANGUAGES C CXX ASM)
            list(APPEND CMAKE_MODULE_PATH {snitch_root}/sw/cmake)
            include(SnitchUtilities)
            add_subdirectory({snitch_root}/sw/snRuntime snRuntime)
            add_snitch_executable(axpy_test axpy_test.cpp axpy.cpp)
            target_include_directories(axpy_test PUBLIC ${{SNRUNTIME_INCLUDE_DIRS}})
            target_include_directories(axpy_test PUBLIC {dace_root}/dace/runtime/include/)
            target_compile_definitions(axpy_test PUBLIC __SNITCH__)
        """))
    
    build_dir = generated_dir / 'build'
    test_file = build_dir / 'axpy_test'
    cmake_cache = build_dir / 'CMakeCache.txt'
    
    if Path(cmake_cache).is_file():
        Path(cmake_cache).unlink()
        
    Path(build_dir).mkdir(exist_ok=True)
    
    if args.simulator == 'banshee':
        subprocess.check_call([
            'cmake',
            '..',
            f'-DCMAKE_TOOLCHAIN_FILE={snitch_toolchain}',
            '-DSNITCH_RUNTIME=snRuntime-banshee',
        ], cwd=build_dir)
        
        subprocess.check_call(['cmake', '--build', build_dir, '--verbose'])
        
        subprocess.check_call([
            banshee_path,
            '--no-opt-llvm',
            '--no-opt-jit',
            '--configuration',
            banshee_config,
            test_file
        ], cwd=build_dir)
        
    elif args.simulator == 'verilator':
        subprocess.check_call([
            'cmake',
            '..',
            f'-DCMAKE_TOOLCHAIN_FILE={snitch_toolchain}',
            '-DSNITCH_RUNTIME=snRuntime-cluster',
        ], cwd=build_dir)
        
        subprocess.check_call(['cmake', '--build', build_dir, '--verbose'])
        
        subprocess.check_call([
            verilator_path,
            test_file
        ], cwd=build_dir)
    else:
        raise RuntimeError('Unknown simulator')

