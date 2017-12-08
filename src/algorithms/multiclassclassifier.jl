module MultiClassClassifier

    using Cxx

    setNClasses(o, nClasses::Integer) = icxx"$o.parameter.nClasses = $nClasses;"
    setTraining(o, training)          = icxx"$o.parameter.training = $training;"
    setPrediction(o, prediction)      = icxx"$o.parameter.prediction = $prediction;"

    module Training

        using Cxx

        Batch(::Type{T}) where {T<:Union{Float32,Float64}} = icxx"daal::algorithms::multi_class_classifier::training::Batch<$T>();"
        Batch() = Batch(Float64)
    end # Training

    module Prediction

        using Cxx

        Batch(::Type{T}) where {T<:Union{Float32,Float64}} = icxx"daal::algorithms::multi_class_classifier::prediction::Batch<$T>();"
        Batch() = Batch(Float64)
    end # Prediction
end