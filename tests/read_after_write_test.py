#!/usr/bin/env python
import dace as dp
import numpy as np

W = dp.symbol()


@dp.program
def raw_prog(A, B):
    tmp = dp.define_local([W], A.dtype)

    @dp.map(_[0:W])
    def compute_tmp(i):
        a << A[i]
        b >> tmp[i]
        b = a

    @dp.map(_[0:W])
    def compute_tmp_again(i):
        a << tmp[i]
        b >> tmp[i]
        b = a + a

    @dp.map(_[0:W])
    def compute_output(i):
        a << tmp[i]
        b >> B[i]
        b = a + a


if __name__ == '__main__':
    W.set(3)

    A = dp.ndarray([W])
    B = dp.ndarray([W])

    A[:] = np.mgrid[0:W.get()]
    B[:] = dp.float32(0.0)

    raw_prog(A, B)

    diff = np.linalg.norm(4 * A - B) / W.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
