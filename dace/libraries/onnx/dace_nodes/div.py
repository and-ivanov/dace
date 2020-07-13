import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.pattern_matching import ExpandTransformation
from .. import environments
from dace.transformation.interstate import GPUTransformSDFG


@dace.library.expansion
class ExpandDivCPU(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "sdiv"
        elif dtype == dace.float64:
            func = "ddiv"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        code = "for(int i=0; i<n; i++)\n    _result[i]=_x[i]/_y[i];"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet

#Todo
#@dace.library.expansion
#class ExpandDivCUDAManual(ExpandTransformation):
#    pass


@dace.program
def divop(_x: dace.float32[32], _y: dace.float32[32], _result: dace.float32[32]):
    _result[:] = _x / _y

@dace.library.expansion
class ExpandDivCUDA(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = divop.to_sdfg()
        sdfg.apply_transformations(GPUTransformSDFG)
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandDivCUDA.make_sdfg(node, state, sdfg)


@dace.library.node
class Div(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "CPU": ExpandDivCPU,
        "GPU": ExpandDivCUDA,
    }
    default_implementation = "CPU"

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_result"},
                         **kwargs)
        self.dtype = dtype


    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to Div")
        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from Div")
        out_memlet = out_edges[0].data
        size = in_memlets[0].subset.size()
        veclen = in_memlets[0].veclen
        if len(size) != 1:
            raise ValueError(
                "Div only supported on 1-dimensional arrays")
        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to Div must have equal size")
        if size != out_memlet.subset.size():
            raise ValueError("Input and Output to Div must have equal size")
        if veclen != in_memlets[1].veclen:
            raise ValueError(
                "Vector lengths of inputs to Div must be identical")
        if (in_memlets[0].wcr is not None or in_memlets[1].wcr is not None
                or out_memlet.wcr is not None):
            raise ValueError("WCR on Div memlets not supported")
