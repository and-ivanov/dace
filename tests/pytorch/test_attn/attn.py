import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
import onnxruntime as ort

import dace
from dace.frontend.onnx import ONNXModel
from dace.frontend.pytorch import DACEModule

from dace.transformation.pattern_matching import match_pattern
from dace.transformation.dataflow.gpu_transform import GPUTransformMap
from dace.transformation.dataflow.map_fusion import MapFusion
from dace.sdfg.graph import SubgraphView

from dace.transformation.subgraph import SubgraphFusion
import dace.transformation.subgraph.helpers as helpers

B = 2
#H = 16
#P = 64
H = 4
P = 16
N = P*H
#SM, SN = 512, 512
SM, SN = 8, 8
#K, Q, V = (torch.randn([SM, B, N], requires_grad=True).cuda(),
#           torch.randn([SN, B, N], requires_grad=True).cuda(),
#           torch.randn([SM, B, N], requires_grad=True).cuda())
K, Q, V = (torch.randn([SM, B, N]),
           torch.randn([SN, B, N]),
           torch.randn([SM, B, N]))

ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)

############################## Torch ################################
pt_outputs = ptmodel(Q, K, V)
print(pt_outputs[0])

############################ DaCe #################################
my_dace_model = DACEModule(ptmodel)
#my_dace_model = DACEModule(ptmodel, (Q, K, V))

my_dace_model.initialize_sdfg((Q, K, V))

my_dace_model.sdfg.save('attn1.sdfg')

my_dace_model.sdfg.expand_library_nodes()

my_dace_model.sdfg.save('attn2.sdfg')

# fuse maps
for sdfg in my_dace_model.sdfg.sdfg_list:
    if sdfg.label == 'softmaxExpansion':
        sdfg_state = sdfg.states()[0]
        SubgraphFusion(sdfg_state).apply(sdfg)

my_dace_model.sdfg.save('attn3.sdfg')

# move single fused map to gpu
for sdfg in my_dace_model.sdfg.sdfg_list:
    if sdfg.label == 'softmaxExpansion':

        matches = match_pattern(sdfg.states()[0], GPUTransformMap, sdfg)
        for m in matches:
            if 'GPUTransformMap in outer_fused[i=0:8, j=0:8]' == m.print_match(sdfg.sdfg_list[m.sdfg_id]):
                m.apply(sdfg)

my_dace_model.sdfg.save('attn4.sdfg')

#TODO: it is possible to explore another path of vectorization from here (along axis in outer node)
# from dace.transformation.dataflow.tiling import MapTiling
#
# for sdfg in my_dace_model.sdfg.sdfg_list:
#     if sdfg.label == 'softmaxExpansion':
#         matches = match_pattern(sdfg.states()[0], MapTiling, sdfg)
#         for m in matches:
#             print(m.print_match(sdfg.sdfg_list[m.sdfg_id]))

from dace.transformation.dataflow.vectorization import Vectorization

for sdfg in my_dace_model.sdfg.sdfg_list:
    if sdfg.label == 'softmaxExpansion':
        matches = match_pattern(sdfg.states()[0], Vectorization, sdfg)
        for m in matches:
            if 'Vectorization in 3 -> 8 -> 4' == m.print_match(sdfg.sdfg_list[m.sdfg_id]):
                m.vector_len = 4
                m.apply(sdfg)
            if 'Vectorization in 9 -> 11 -> 10' == m.print_match(sdfg.sdfg_list[m.sdfg_id]):
                m.vector_len = 4
                m.apply(sdfg)

my_dace_model.sdfg.save('attn5.sdfg')

for sdfg in my_dace_model.sdfg.sdfg_list:
    if sdfg.label == 'softmaxExpansion':
        matches = match_pattern(sdfg.states()[0], Vectorization, sdfg)
        for m in matches:
            if m.print_match(sdfg.sdfg_list[m.sdfg_id]) == 'Vectorization in 0 -> 2 -> 1':
                m.apply(sdfg.sdfg_list[m.sdfg_id])

my_dace_model.sdfg.save('attn6.sdfg')

from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.dataflow.trivial_map_range_elimination import TrivialMapRangeElimination

for sdfg in my_dace_model.sdfg.sdfg_list:
    for transform in [TrivialMapElimination, TrivialMapRangeElimination]:
        matches = match_pattern(sdfg.states()[0], transform, sdfg)
        for m in matches:
            m.apply(sdfg.sdfg_list[m.sdfg_id])

my_dace_model.sdfg.save('attn6_1.sdfg')

from dace.transformation.dataflow.tiling import MapTiling

for sdfg in my_dace_model.sdfg.sdfg_list:
    if sdfg.label == 'softmaxExpansion':
        matches = match_pattern(sdfg.states()[0], MapTiling, sdfg)
        for m in matches:
            if "MapTiling in reduce_values: ['_i2']" == m.print_match(sdfg.sdfg_list[m.sdfg_id]):
                m.tile_sizes = (4,)
                m.apply(sdfg.sdfg_list[m.sdfg_id])

my_dace_model.sdfg.save('attn6_2.sdfg')

# TODO:
# 1. introduce accessnode (of size 1) between map exits (basically this transformation is an adaptation of WCRExtraction)
# 2. apply WCRExtraction + Vectorization to the inner map
# 3. design new transformation WarpAllReduceDetection that finds patterns AccessNode->WCR->MapExit(trheads)->WCR->AccessNode
# and replaces it by AccessNode->WarpReduceTasklet->MapExit(threads)->AccessNode

from dace.transformation.dataflow.stream_transient import AccumulateTransient

for sdfg in my_dace_model.sdfg.sdfg_list:
    if sdfg.label == 'softmaxExpansion':
        matches = match_pattern(sdfg.states()[0], AccumulateTransient, sdfg)
        for m in matches:
            print(m.print_match(sdfg.sdfg_list[m.sdfg_id]))
            if "AccumulateTransient in 2 -> 1 -> 6" == m.print_match(sdfg.sdfg_list[m.sdfg_id]):
                m.apply(sdfg.sdfg_list[m.sdfg_id])

my_dace_model.sdfg.save('attn7.sdfg')

# from dace.transformation.dataflow.wcr_extraction import WCRExtraction
#
# for sdfg in my_dace_model.sdfg.sdfg_list:
#     if sdfg.label == 'softmaxExpansion':
#         matches = match_pattern(sdfg.states()[0], WCRExtraction, sdfg)
#         for m in matches:
#             print(m.print_match(sdfg.sdfg_list[m.sdfg_id]))
#             m.apply(sdfg.sdfg_list[m.sdfg_id])


my_dace_model.sdfg.save('attn8.sdfg')

dace_outputs = my_dace_model(Q, K, V)

#dace_outputs = my_dace_model(Q.numpy(), K.numpy(), V.numpy()) 
print(dace_outputs[0])

############################## Ort #################################
sess = ort.InferenceSession("shape_infer.onnx")
ort_outputs = sess.run(None, {"actual_input_1": Q.numpy(), "key": K.numpy(), "value": V.numpy()})
print(ort_outputs[0])



assert np.allclose(dace_outputs[0], ort_outputs[0], rtol=1e-02)
assert np.allclose(dace_outputs[1], ort_outputs[1], rtol=1e-02)
assert np.allclose(pt_outputs[0].detach().numpy(), ort_outputs[0], rtol=1e-02)
assert np.allclose(pt_outputs[1].detach().numpy(), ort_outputs[1], rtol=1e-02)

print("Test passed.")
