import FFTW
import LinearAlgebra

# set to zero Fourier coefficients
function _apply_mask(û::SpectralField{P, LD, L}) where {P, LD, L}
    for l = L+1:LD+1
        @inbounds @simd for p = 1:P+1
            û.data[p, l+1] = 0
        end
    end
    return û
end

export ForwardFFT!, InverseFFT!

struct ForwardFFT!{P, LD, PLAN}
    plan::PLAN
    function ForwardFFT!(        u::PhysicalField{P, LD},
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {P, LD}
        plan = FFTW.plan_rfft(similar(parent(u)), [2]; flags=flags, timelimit=timelimit)
        return new{P, LD, typeof(plan)}(plan)
    end
end

function (f::ForwardFFT!{P, LD})(û::SpectralField{P, L, LD}, u::PhysicalField{P, LD}) where {P, L, LD}
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    û .*= 1/(2*LD+2)
    return _apply_mask(û)
end


struct InverseFFT!{P, LD, PLAN}
    plan::PLAN
    function InverseFFT!(        û::SpectralField{P, L, LD}, 
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {P, L, LD}
        plan = FFTW.plan_brfft(similar(parent(û)), 2LD+2, [2]; flags=flags, timelimit=timelimit)
        return new{P, LD, typeof(plan)}(plan)
    end
end

function (i::InverseFFT!{P, LD})(u::PhysicalField{P, LD}, û::SpectralField{P, L, LD}) where {P, L, LD}
    FFTW.unsafe_execute!(i.plan, parent(_apply_mask(û)), parent(u))
    return u
end