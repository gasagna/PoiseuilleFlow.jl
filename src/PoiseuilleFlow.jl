module PoiseuilleFlow

using LoopVectorization

include("physicalfield.jl")
include("spectralfield.jl")
include("ffts.jl")
include("chebutils.jl")
include("equations.jl")
include("operators.jl")
include("banded.jl")

end