module NeuralNetworks
    module Layers
        addNext() = error("")
        getLayerInput() = error("")
        getLayerParameter() = error("")
        set() = error("")

        module Forward
            import ....Algorithms: Parameter
            import ..Layers: addNext, getLayerInput, getLayerParameter, set

            using Cxx

            const Biases  = icxx"daal::algorithms::neural_networks::layers::forward::biases;"
            const Weights = icxx"daal::algorithms::neural_networks::layers::forward::weights;"

            struct Input
                o::Cxx.CppPtr
            end

            struct LayerDescriptor
                o::Cxx.CppRef
            end

            addNext(o::LayerDescriptor, index::Integer) = icxx"$(o.o).addNext($index);"

            struct LayerIfacePtr
                o::Cxx.CppValue
            end

            getLayerInput(    o::LayerIfacePtr) = Input(icxx"$(o.o)->getLayerInput();")
            getLayerParameter(o::LayerIfacePtr) = Parameter(icxx"$(o.o)->getLayerParameter();")

            set(o::Input, id, ptr) = icxx"$(o.o)->set($id, $ptr);"
        end # Forward

        module FullyConnected
            module Forward
                using Cxx

                struct Batch{T<:Union{Float32,Float64}}
                    o::Cxx.CppValue
                end

                Batch(::Type{T}, nOutputs::Integer) where {T<:Union{Float32,Float64}} =
                    Batch{T}(icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::layers::fullyconnected::forward::Batch<$T>>(new      daal::algorithms::neural_networks::layers::fullyconnected::forward::Batch<$T>($nOutputs));")
                Batch(nOutputs::Integer) = Batch(Float64, nOutputs)
            end # Forward
        end # FullyConnected

        module Softmax
            module Forward
                using Cxx

                struct Batch{T<:Union{Float32,Float64}}
                    o::Cxx.CppValue
                end

                Batch(::Type{T}) where {T<:Union{Float32,Float64}} =
                    Batch{T}(icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::layers::softmax::forward::Batch<$T>>(new     daal::algorithms::neural_networks::layers::softmax::forward::Batch<$T>());")
                Batch() = Batch(Float64)
            end # Forward
        end # Softmax
    end # Layers

    module Prediction
        import Base: get
        import ....Daal.DataManagement: Tensor
        import ...Algorithms: Parameter, add
        import ..NeuralNetworks.Layers.Forward: LayerDescriptor, LayerIfacePtr

        using Cxx

        const DataId       = icxx"daal::algorithms::neural_networks::prediction::data;"
        const ModelId      = icxx"daal::algorithms::neural_networks::prediction::model;"
        const PredictionId = icxx"daal::algorithms::neural_networks::prediction::prediction;"

        struct Topology
            o::Cxx.CppValue
        end
        struct TopologyPtr
            o::Cxx.CppValue
        end

        Topology()    = Topology(icxx"daal::algorithms::neural_networks::prediction::Topology();")
        TopologyPtr() = TopologyPtr(icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::prediction::Topology>(new daal::algorithms::neural_networks::prediction::Topology());")

        add(o::TopologyPtr, value) = icxx"$(o.o)->add($(value.o));"

        get(o::TopologyPtr, id) = LayerDescriptor(icxx"$(o.o)->get($id);")

        struct Model
            o::Cxx.CppValue
        end
        struct ModelPtr
            o::Cxx.CppValue
        end

        ModelPtr(o::TopologyPtr) = ModelPtr(icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::prediction::Model>(new daal::algorithms::neural_networks::prediction::Model(*$(o.o)));")

        # FixMe! According to Keno it shouldn't be necessary to use a temporary variable for
        # this but the reference count in the SharedPtr isn't correct if I leave it out. Might
        # be an issue with Cxx.jl
        getLayer(o::ModelPtr, index::Integer) = LayerIfacePtr(icxx"daal::algorithms::neural_networks::layers::forward::LayerIfacePtr ptr = $(o.o)->getLayer($index); ptr;")

        struct Batch{T<:Union{Float32,Float64}}
            o::Cxx.CppValue
        end
        Batch(::Type{T}) where {T<:Union{Float32,Float64}} = Batch{T}(icxx"daal::algorithms::neural_networks::prediction::Batch<$T>();")
        Batch() = Batch(Float64)

        # FixMe! Should eventually become a generated function
        for p in (:input, :parameter)
            @eval begin
                qp = $(QuoteNode(p))
                Base.getindex(o::Batch, ::Val{qp}) = Parameter(@icxx_str $"&\$(o.o).$p;")
            end
        end

        struct Result
            o::Cxx.CppValue
        end

        getResult(o::Batch) = Result(icxx"daal::services::SharedPtr<daal::algorithms::neural_networks::prediction::Result> ptr = $(o.o).getResult(); ptr;")

        get(o::Result, id) = Tensor(icxx"daal::data_management::TensorPtr ptr = $(o.o)->get($id); ptr;")
    end # Prediction
end # NeuralNetworks
