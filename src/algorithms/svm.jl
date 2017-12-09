module SVM

    using Cxx

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