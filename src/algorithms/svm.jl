module SVM

    using Cxx

    # Base.getindex(o::CxxType, Val{:parameter}) = icxx"$o.->parameter;"
    # Base.setindex!(o::CxxType, kernel, Val{:kernel}) = icxx"($(o.o))->parameter.kernel = $(kernel.o);"

    setKernel(o, kernel) = icxx"$o->parameter.kernel = $kernel;"
    setCacheSize(o, sz::Integer) = icxx"$o->parameter.cacheSize = $sz;"

    module Training

        using Cxx

        Batch(::Type{T}) where {T<:Union{Float32,Float64}} = icxx"daal::services::SharedPtr<daal::algorithms::svm::training::Batch<$T>>(new daal::algorithms::svm::training::Batch<$T>());"
        Batch() = Batch(Float64)
    end # Training

    module Prediction

        using Cxx

        Batch(::Type{T}) where {T<:Union{Float32,Float64}} = icxx"daal::services::SharedPtr<daal::algorithms::svm::prediction::Batch<$T>>(new daal::algorithms::svm::prediction::Batch<$T>());"
        Batch() = Batch(Float64)
    end # Prediction
end