module NeuralNetworks
    module Layers

        using Cxx

        getLayerInput() = error("")
        getLayerParameter() = error("")

        addNext(o, index::Integer) = icxx"$o.addNext($index);"

        getLayerInput(    o) = icxx"$o->getLayerInput();"
        getLayerParameter(o) = icxx"$o->getLayerParameter();"

        module Forward

            using Cxx

            const Biases  = icxx"daal::algorithms::neural_networks::layers::forward::biases;"
            const Weights = icxx"daal::algorithms::neural_networks::layers::forward::weights;"
        end # Forward

        module FullyConnected
            module Forward
                using Cxx

                Batch(::Type{T}, nOutputs::Integer) where {T<:Union{Float32,Float64}} =
                    icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::layers::fullyconnected::forward::Batch<$T>>(new      daal::algorithms::neural_networks::layers::fullyconnected::forward::Batch<$T>($nOutputs));"
                Batch(nOutputs::Integer) = Batch(Float64, nOutputs)
            end # Forward
        end # FullyConnected

        module Softmax
            module Forward
                using Cxx

                Batch(::Type{T}) where {T<:Union{Float32,Float64}} =
                    icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::layers::softmax::forward::Batch<$T>>(new     daal::algorithms::neural_networks::layers::softmax::forward::Batch<$T>());"
                Batch() = Batch(Float64)
            end # Forward
        end # Softmax
    end # Layers

    module Prediction

        using Cxx

        const DataId       = icxx"daal::algorithms::neural_networks::prediction::data;"
        const ModelId      = icxx"daal::algorithms::neural_networks::prediction::model;"
        const PredictionId = icxx"daal::algorithms::neural_networks::prediction::prediction;"

        Topology()    = icxx"daal::algorithms::neural_networks::prediction::Topology();"
        TopologyPtr() = icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::prediction::Topology>(new daal::algorithms::neural_networks::prediction::Topology());"

        # add(o::TopologyPtr, value) = icxx"$(o.o)->add($(value.o));"

        # get(o::TopologyPtr, id) = LayerDescriptor(icxx"$(o.o)->get($id);")

        ModelPtr(o) = icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::prediction::Model>(new daal::algorithms::neural_networks::prediction::Model(*$o));"

        # FixMe! According to Keno it shouldn't be necessary to use a temporary variable for
        # this but the reference count in the SharedPtr isn't correct if I leave it out. Might
        # be an issue with Cxx.jl
        # getLayer(o::ModelPtr, index::Integer) = LayerIfacePtr(icxx"daal::algorithms::neural_networks::layers::forward::LayerIfacePtr ptr = $(o.o)->getLayer($index); ptr;")

        # getLayer(o, index::Integer) = icxx"$o->getLayer($index);"
        getLayer(o, index::Integer) = icxx"daal::algorithms::neural_networks::layers::forward::LayerIfacePtr ptr = $o->getLayer($index); ptr;"

        Batch(::Type{T}) where {T<:Union{Float32,Float64}} = icxx"daal::algorithms::neural_networks::prediction::Batch<$T>();"
        Batch() = Batch(Float64)

        # getResult(o::Batch) = Result(icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::prediction::Result> ptr = $(o.o).getResult(); ptr;")

        # get(o::Result, id) = Tensor(icxx"daal::data_management::TensorPtr ptr = $(o.o)->get($id); ptr;")
    end # Prediction
end # NeuralNetworks
