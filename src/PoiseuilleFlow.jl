module PoiseuilleFlow

using LoopVectorization

include("field.jl")
include("ftfield.jl")
include("ffts.jl")
include("chebutils.jl")
include("equations.jl")
include("operators.jl")
include("banded.jl")

end