module KernelFunction

    module Linear

        using Cxx

        struct FastCSR end

        Batch(::Type{T}, ::FastCSR) where {T<:Union{Float32,Float64}} =
            icxx"daal::services::SharedPtr<daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>>(new daal::algorithms::kernel_function::linear::Batch<$T,daal::algorithms::kernel_function::linear::fastCSR>());"
    end # Linear
end # KernelFunction