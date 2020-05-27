using BenchmarkTools
using LinearAlgebra
using PoiseuilleFlow
using Printf
using PyPlot; pygui(true)

t_ul_s = []
t_solve_s = []
Ns = 5:500

for N = Ns
    b = rand(N)
    l = rand(N-1)
    d = rand(N-1)
    u = rand(N-2)
    c = rand(N)
    A = QuasiTridiagonal(b, l, d, u)
    t_ul = minimum(@elapsed ul!(A) for i = 1:1000)
    t_solve = minimum(@elapsed ldiv!(A, c) for i = 1:1000)
    push!(t_ul_s, t_ul)
    push!(t_solve_s, t_solve)
    @printf "%04d %.6f\n" N 10^9*t_solve / N
end

figure(1)
plot(Ns, 10^9*t_ul_s ./ Ns, label="factorise" )
plot(Ns, 10^9*t_solve_s ./ Ns, label="solve" )
xlabel("N")
ylabel("t/N [ns]")
show()