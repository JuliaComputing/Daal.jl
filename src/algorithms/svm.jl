module SVM

using Cxx

abstract type Batch end

setKernel(o::Batch, kernel) = icxx"($(o.o))->parameter.kernel = $(kernel.o);"
setCacheSize(o::Batch, sz::Integer) = icxx"($(o.o))->parameter.cacheSize = $sz;"

# There are two different SVM methods so we might want to add a method parameter at some point
struct TrainingBatch{T<:Union{Float32,Float64}} <: Batch
    o::Cxx.CppValue
end

TrainingBatch(::Type{T}) where {T<:Union{Float32,Float64}} = TrainingBatch{T}(icxx"daal::services::SharedPtr<daal::algorithms::svm::training::Batch<$T>>(new daal::algorithms::svm::training::Batch<$T>());")
TrainingBatch() = TrainingBatch(Float64)

# There are two different SVM methods so we might want to add a method parameter at some point
struct PredictionBatch{T<:Union{Float32,Float64}} <: Batch
    o::Cxx.CppValue
end

PredictionBatch(::Type{T}) where {T<:Union{Float32,Float64}} = PredictionBatch{T}(icxx"daal::services::SharedPtr<daal::algorithms::svm::prediction::Batch<$T>>(new daal::algorithms::svm::prediction::Batch<$T>());")
PredictionBatch() = PredictionBatch(Float64)

end