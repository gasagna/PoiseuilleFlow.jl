using PoiseuilleFlow
using BenchmarkTools

P, L, Lx = 127, 127, 2π

 f = SpectralField(P, L, Lx)
df = SpectralField(P, L, Lx)

# @btime (ddy!($df, $f); ddy!($df, $f))
# @btime d2dy2!($df, $f)

@btime ddx!($df, $f)