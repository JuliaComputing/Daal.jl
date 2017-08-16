module MKLVSL

include("mkl_vsl_defines.jl")

const libmkl = Base.libblas_name

using Base.LinAlg: BlasInt

abstract type VSLRNG <: AbstractRNG end

function delete(stream::VSLRNG)
    err = ccall((:vslDeleteStream, libmkl), Cint, (Ptr{Ptr{Void}},), &stream.ptr)
    if err != 0
        error()
    end
    return nothing
end

# RNGs
for t in (
    :MCG31,
    :R250,
    :MRG32K3A,
    :MCG59,
    :WH,
    :SOBOL,
    :NIEDERR,
    :MT19937,
    :MT2203,
#    :IABSTRACT,
#    :DABSTRACT,
#    :SABSTRACT,
    :SFMT19937,
    :NONDETERM,
    :ARS5,
    :PHILOX4X32X10)

    vsl_enum = Symbol(:VSL_BRNG_, t)
    @eval begin
        type $t <: VSLRNG
            ptr::Ptr{Void}
            function $t(seed::Integer = rand(Base.RandomDevice(), Int32))
                s = Ref{Ptr{Void}}()
                err = ccall((:vslNewStream, libmkl), Cint,
                    (Ref{Ptr{Void}}, BlasInt, BlasInt),
                        s, $vsl_enum, seed)
                if err != 0
                    error()
                end
                rng = new(s[])
                finalizer(rng, delete)
                return rng
            end
        end
    end
end

# rand! and randn!
for (s, elty) in ((:vsRngUniform, :Float32), (:vdRngUniform, :Float64))
    @eval begin
        function Base.Random.rand!(s::VSLRNG, x::Array{$elty})
            err = ccall(($(string(s)), libmkl), Cint,
                (BlasInt, Ptr{Void}, BlasInt, Ptr{$elty}, $elty, $elty),
                    VSL_RNG_METHOD_UNIFORM_STD, s.ptr, length(x), x, 0.0, 1.0)
            if err != 0
                error()
            end
            return x
        end
    end
end

for (s, elty) in ((:vsRngGaussian, :Float32), (:vdRngGaussian, :Float64))
    @eval begin
        function Base.Random.randn!(s::VSLRNG, x::Array{$elty})
            err = ccall(($(string(s)), libmkl), Cint,
                (BlasInt, Ptr{Void}, BlasInt, Ptr{$elty}, $elty, $elty),
                    VSL_RNG_METHOD_UNIFORM_STD, s.ptr, length(x), x, 0.0, 1.0)
            if err != 0
                error()
            end
            return x
        end
    end
end

end # module
