export Grid, points, weights, domain

struct Grid{P, L, LD, DT, XY, W}
    Ds::DT # tuple of differentation matrices
    xy::XY # grid points
     w::W  # weights for numerical integration 
   dom::NTuple{2, Float64}
    function Grid(P::Int, L::Int, LD::Int, Lx::Real; η::Real=0.5, width::Int=7) 
        # actual points
        y = FDGrids.gridpoints(P, -1, 1, η)
        x = (0:(2LD+1))/(2LD+2)*Lx
        xy = (x, y)

        # matrices for the time stepping
        Ds = (Multiplier(FDGrids.DiffMatrix(y, width, 1)),
              Multiplier(FDGrids.DiffMatrix(y, width, 2)),
              Multiplier(FDGrids.DiffMatrix(y, width, 4)))

        # weights for quadrature (trapz for now)
        dy = diff(y)
        w = [[dy[i]/2 for i = 1:P-1]..., 0] + [0, [dy[i-1]/2 for i = 2:P]...]

        return new{P, L, LD, typeof(Ds), typeof(xy), typeof(w)}(Ds, xy, w, (Lx, 2))
    end
end

Base.size(g::Grid{P, L, LD}, which::Symbol) where {P, L, LD} = 
    (which == :spectral ? (P, L)     :
     which == :physical ? (P, 2LD+2) : throw(ArgumentError("invalid argument: $which")))

# domain size
domain(g::Grid) = g.dom

# get points
points(g::Grid) = g.xy

# get weights
weights(g::Grid) = g.w