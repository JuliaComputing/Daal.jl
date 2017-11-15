module MultiClassClassifier

import ..Algorithms: compute
import ..Algorithms.Classifier.PredictionResult

using Cxx

abstract type Batch end

setNClasses(o::Batch, nClasses::Integer) = icxx"$(o.o).parameter.nClasses = $nClasses;"
setTraining(o::Batch, training)          = icxx"$(o.o).parameter.training = $(training.o);"
setPrediction(o::Batch, prediction)      = icxx"$(o.o).parameter.prediction = $(prediction.o);"

# Training
struct TrainingBatch{T<:Union{Float32,Float64}} <: Batch# there is only a single method for the class so we omit it
    o::Cxx.CppValue
end

TrainingBatch(::Type{T}) where {T<:Union{Float32,Float64}} = TrainingBatch{T}(icxx"daal::algorithms::multi_class_classifier::training::Batch<$T>();")
TrainingBatch() = TrainingBatch(Float64)

struct TrainingResult
    o::Cxx.CppValue
end
# TrainingResult() = TrainingResult(icxx"daal::services::SharedPtr<daal::algorithms::multi_class_classifier::training::Result>(new daal::algorithms::multi_class_classifier::training::Result());")

getResult(o::TrainingBatch) = TrainingResult(icxx"$(o.o).getResult();")

struct Model
    o::Cxx.CppValue
end

get(o::TrainingResult, id) = Model(icxx"$(o.o)->get($id);")

# SharedPtr(::Type{TrainingResult}) = SharedPtr{TrainingResult}(icxx"daal::services::SharedPtr<daal::algorithms::multi_class_classifier::training::Result>(new daal::algorithms::multi_class_classifier::training::Result());")

# Prediction
struct PredictionBatch{T<:Union{Float32,Float64}} <: Batch # there is only a single method for the class so we omit it
    o::Cxx.CppValue
end

PredictionBatch(::Type{T}) where {T<:Union{Float32,Float64}} = PredictionBatch{T}(icxx"daal::algorithms::multi_class_classifier::prediction::Batch<$T>();")
PredictionBatch() = PredictionBatch(Float64)

# struct PredictionResult
    # o::Cxx.CppValue
# end

getResult(o::PredictionBatch) = PredictionResult(icxx"$(o.o).getResult();")

end