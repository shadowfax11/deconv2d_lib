function s = shrinkageOp(z,lmbd)

s = max(abs(z)-lmbd,0).*sign(z);