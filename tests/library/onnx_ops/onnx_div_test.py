import dace
from dace.memlet import Memlet
from dace.codegen.compiler import CompilerConfigurationError, CompilationError
import dace.libraries.onnx as daceonnx
#import dace.libraries.blas as blas
import numpy as np
import sys
import warnings

###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("div_{}_{}".format(implementation, dtype))
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x" + suffix, [n],
                   dtype,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("y" + suffix, [n],
                   dtype,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("result" + suffix, [n],
                   dtype,
                   storage=storage,
                   transient=transient)

    x = state.add_read("x" + suffix)
    y = state.add_read("y" + suffix)
    result = state.add_write("result" + suffix)

    div_node = daceonnx.div.Div("div", dtype)
    #div_node = blas.div.Div("div", dtype)
    div_node.implementation = implementation

    state.add_memlet_path(x,
                          div_node,
                          dst_conn="_x",
                          memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(y,
                          div_node,
                          dst_conn="_y",
                          memlet=Memlet.simple(y, "0:n", num_accesses=n))
    # TODO: remove -1 once this no longer triggers a write in the codegen.
    state.add_memlet_path(div_node,
                          result,
                          src_conn="_result",
                          memlet=Memlet.simple(result, "0:n", num_accesses=n))

    if storage != dace.StorageType.Default:

        sdfg.add_array("x", [n], dtype)
        sdfg.add_array("y", [n], dtype)
        sdfg.add_array("result", [n], dtype)

        init_state = sdfg.add_state("copy_to_device")
        sdfg.add_edge(init_state, state, dace.InterstateEdge())

        x_host = init_state.add_read("x")
        y_host = init_state.add_read("y")
        x_device = init_state.add_write("x" + suffix)
        y_device = init_state.add_write("y" + suffix)
        init_state.add_memlet_path(x_host,
                                   x_device,
                                   memlet=Memlet.simple(x_host,
                                                        "0:n",
                                                        num_accesses=n))
        init_state.add_memlet_path(y_host,
                                   y_device,
                                   memlet=Memlet.simple(y_host,
                                                        "0:n",
                                                        num_accesses=n))

        finalize_state = sdfg.add_state("copy_to_host")
        sdfg.add_edge(state, finalize_state, dace.InterstateEdge())

        result_device = finalize_state.add_write("result" + suffix)
        result_host = finalize_state.add_read("result")
        finalize_state.add_memlet_path(result_device,
                                       result_host,
                                       memlet=Memlet.simple(result_device,
                                                            "0:n",
                                                            num_accesses=n))

    return sdfg


###############################################################################


def _test_div(implementation, dtype, sdfg):
    #try:
    #    div = sdfg.compile()
    #except (CompilerConfigurationError, CompilationError):
    #    warnings.warn(
    #        'Configuration/compilation failed, library missing or '
    #        'misconfigured, skipping test for {}.'.format(implementation))
    #    return

    div = sdfg
    size = 32

    x = np.ndarray(size, dtype=dtype)
    y = np.ndarray(size, dtype=dtype)
    result = np.ndarray(size, dtype=dtype)

    x[:] = 4.0
    y[:] = 2.0
    result[:] = 0.0

    div(x=x, y=y, result=result, n=size)

    ref = x/y
    print("ref: ", ref)
    print("result: ", result)
    assert np.allclose(ref, result)
    print("Test ran successfully for {}.".format(implementation))


def test_div():
    _test_div("32-bit CPU", np.float32, make_sdfg("CPU", dace.float32))
    _test_div("32-bit GPU", np.float32, make_sdfg("GPU", dace.float32, dace.StorageType.GPU_Global))


###############################################################################

if __name__ == "__main__":
    test_div()
###############################################################################
