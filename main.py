import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.optimize import minimize
import numpy
mod=None
with open("./print.cu","r") as f:
    mod=SourceModule(f.read())
# 从编译的函数中取得函数
print_cu=mod.get_function("print_cu")
# 在python中定义numpy的数组
a=numpy.random.randn(4,4)
# cuda仅支持float
a=a.astype(numpy.float32)
# 在cuda中分配a大小的空间
a_gpu=cuda.mem_alloc(a.nbytes)
# 将python中申请的数据放到cuda申请的内存中
cuda.memcpy_htod(a_gpu,a)
print_cu(a_gpu,block=(1,1,10))
a_doubled=numpy.empty_like(a)
# 这里是真正的运行，在隐含的运行前面的函数，随后从cuda内存取数据到python空间
cuda.memcpy_dtoh(a_doubled,a_gpu)