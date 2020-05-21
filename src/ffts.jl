import FFTW
import LinearAlgebra

export ForwardFFT!, InverseFFT!

struct ForwardFFT!{P, L, PLAN}
    plan::PLAN
    function ForwardFFT!(        u::Field{P, L},
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {P, L}
        plan = FFTW.plan_rfft(parent(u), [2]; flags=flags, timelimit=timelimit)
        return new{P, L, typeof(plan)}(plan)
    end
end

function (f::ForwardFFT!{P, L})(û::FTField{P, L}, u::Field{P, L}) where {P, L}
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    û .*= 1/(2*L+1)
    return û
end


struct InverseFFT!{P, L, PLAN}
    plan::PLAN
    function InverseFFT!(        û::FTField{P, L}, 
                             flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {P, L}
        plan = FFTW.plan_brfft(parent(û), 2L+1, [2]; flags=flags, timelimit=timelimit)
        return new{P, L, typeof(plan)}(plan)
    end
end

function (i::InverseFFT!{P, L})(u::Field{P, L}, û::FTField{P, L}) where {P, L}
    FFTW.unsafe_execute!(i.plan, parent(û), parent(u))
    return u
end