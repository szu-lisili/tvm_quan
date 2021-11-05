import tvm
from tvm import relay
from tvm import autotvm
import tvm.contrib.util as util
import numpy as np
tgt = tvm.target.Target(target="llvm", host="llvm")


x = relay.var('x', shape=[1, 224, 224, 32], dtype='int16')
#w = relay.var('w', shape=[32, 32, 3, 3], dtype='int16')
#w_np = np.random.normal(size=[32, 32, 3, 3]).astype('float32').astype('int16')
w = relay.var('w', shape=[1, 1, 32, 32], dtype='int16')
#w = relay.var('w', dtype='int16')
w_np = np.random.normal(size=[1, 1, 32, 32]).astype('int16')
x_np = np.random.normal(size=[1, 224, 224, 32]).astype('int16')

#y = relay.nn.conv2d(x, w, data_layout='NHWC', kernel_size=[3, 3], kernel_layout='OIHW')
bpw = relay.nn.bitpack(w, bit_axis=2, pack_axis=2, bits=1, pack_type='uint8')
#bpw = relay.nn.bitpack(w, bits=1, pack_axis=2, bit_axis=4, pack_type='uint8')
#y = relay.nn.bitserial_conv2d(x, bpw, data_layout='NHWC', kernel_size=[1, 1], channels=32, pack_dtype='uint8', kernel_layout='HWIO')
y = relay.nn.conv2d(x, w, data_layout='NHWC', kernel_size=[1, 1], channels=32, kernel_layout='HWIO')
y_func = relay.Function([x, w], y)

print(y_func.body)
params = {'w': w_np}
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(y_func, target=target, params=params)


tmp = util.tempdir()
lib_fname = tmp.relpath('net.tar')
lib.export_library(lib_fname)