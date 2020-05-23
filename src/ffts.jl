import FFTW
import LinearAlgebra

export ForwardFFT!, InverseFFT!

struct ForwardFFT!{P, L, PLAN_X, PLAN_Y}
    plan_x::PLAN_X
    plan_y::PLAN_Y
    function ForwardFFT!(        u::PhysicalField{P, L},
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {P, L}
        plan_x = FFTW.plan_rfft(parent(u), [2]; flags=flags, timelimit=timelimit)
        plan_y = FFTW.plan_r2r!(parent(u), FFTW.REDFT00, [1]; flags=flags, timelimit=timelimit)
        return new{P, L, typeof(plan_x), typeof(plan_y)}(plan_x, plan_y)
    end
end

function (f::ForwardFFT!{P, L})(û::SpectralField{P, L}, u::PhysicalField{P, L}) where {P, L}
    FFTW.unsafe_execute!(f.plan_y, parent(u), parent(u))
    FFTW.unsafe_execute!(f.plan_x, parent(u), parent(û))
    û .*= 1/((2*L+1) * 2*P)
    return û
end


struct InverseFFT!{P, L, PLAN_X, PLAN_Y}
    plan_x::PLAN_X
    plan_y::PLAN_Y
    function InverseFFT!(        û::SpectralField{P, L, Lx, T},
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {P, L, Lx, T}
        plan_x = FFTW.plan_brfft(parent(û), 2L+1, [2]; flags=flags, timelimit=timelimit)
        plan_y = FFTW.plan_r2r!(zeros(T, P+1, 2*L+1), FFTW.REDFT00, [1]; flags=flags, timelimit=timelimit)
        return new{P, L, typeof(plan_x), typeof(plan_y)}(plan_x, plan_y)
    end
end

function (i::InverseFFT!{P, L})(u::PhysicalField{P, L}, û::SpectralField{P, L}) where {P, L}
    FFTW.unsafe_execute!(i.plan_x, parent(û), parent(u))
    FFTW.unsafe_execute!(i.plan_y, parent(u), parent(u))
    return u
end

