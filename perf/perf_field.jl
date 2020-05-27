using PoiseuilleFlow
using BenchmarkTools

P, L, Lx = 127, 127, 2π

foo(f, g, h) = (f .= g; f)

f̂ = SpectralField(P, L, Lx)
ĝ = SpectralField(P, L, Lx)
ĥ = SpectralField(P, L, Lx)

f = PhysicalField(P, L, Lx)
g = PhysicalField(P, L, Lx)
h = PhysicalField(P, L, Lx)

@btime foo($f, $g, $h)
@btime foo($f̂, $ĝ, $ĥ)