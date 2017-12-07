module KernelFunction

module Linear

using Cxx

struct FastCSR end

struct Batch{T<:Union{Float32,Float64},M}
    o::Cxx.CppValue
end

Batch(::Type{T}, ::FastCSR) where {T<:Union{Float32,Float64}} =
    Batch{T,FastCSR}(icxx"daal::services::SharedPtr<daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>>(new daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>());")

end

end