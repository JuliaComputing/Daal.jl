module Classifier

    module Training

        using Cxx

        DataId   = icxx"daal::algorithms::classifier::training::data;"
        LabelsId = icxx"daal::algorithms::classifier::training::labels;"
        ModelId  = icxx"daal::algorithms::classifier::training::model;"
    end # Training

    module Prediction

        using Cxx

        DataId       = icxx"daal::algorithms::classifier::prediction::data;"
        ModelId      = icxx"daal::algorithms::classifier::prediction::model;"
        PredictionId = icxx"daal::algorithms::classifier::prediction::prediction;"
    end # Prediction
end