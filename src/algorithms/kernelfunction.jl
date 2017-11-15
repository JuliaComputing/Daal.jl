module KernelFunction

module Linear

# import ....Daal: DaalClass, delete
import ....Daal.Services.SharedPtr

using Cxx

struct FastCSR end

struct Batch{T<:Union{Float32,Float64},M}
    o::Cxx.CppValue
end

Batch(::Type{T}, ::FastCSR) where {T<:Union{Float32,Float64}} = Batch{T,FastCSR}(icxx"daal::services::SharedPtr<daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>>(new daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>());")

# SharedPtr(::Type{Batch{T,FastCSR}}) where {T} = SharedPtr{Batch{T,FastCSR}}(icxx"daal::services::SharedPtr<daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>>(new daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>());")

end

end