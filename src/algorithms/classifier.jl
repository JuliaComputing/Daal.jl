module Classifier

import ....Daal.DataManagement: NumericTable

using Cxx

# Training

TrainingData   = icxx"daal::algorithms::classifier::training::data;"
TrainingLabels = icxx"daal::algorithms::classifier::training::labels;"
TrainingModel  = icxx"daal::algorithms::classifier::training::model;"

# Prediction

PredictionData       = icxx"daal::algorithms::classifier::prediction::data;"
PredictionModel      = icxx"daal::algorithms::classifier::prediction::model;"
PredictionPrediction = icxx"daal::algorithms::classifier::prediction::prediction;"

struct PredictionResult
    o
end

get(o::PredictionResult, id) = NumericTable(icxx"$(o.o)->get($id);")

# PredictionResult() = PredictionResult(icxx"daal::algorithms::classifier::prediction::Result();")

# SharedPtr(::Type{PredictionResult}) = SharedPtr{PredictionResult}(icxx"daal::services::SharedPtr<daal::algorithms::classifier::prediction::Result>(new daal::algorithms::classifier::prediction::Result());")

end