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

my_dace_model.sdfg.expand_library_nodes()

my_dace_model.sdfg.save('my_before.sdfg')

for sdfg in my_dace_model.sdfg.sdfg_list:
    if sdfg.label == 'softmaxExpansion':
        matches = match_pattern(sdfg.states()[0], MapFusion, sdfg)
        for m in matches:
            print(m.print_match(sdfg.sdfg_list[m.sdfg_id]))
            m.apply(sdfg)

my_dace_model.sdfg.save('my_after.sdfg')


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
