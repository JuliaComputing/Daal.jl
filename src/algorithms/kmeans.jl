module Kmeans

    import ....Daal: Step2Master, Step1Local

    using Cxx

    Data           = icxx"daal::algorithms::kmeans::data;"
    InputCentroids = icxx"daal::algorithms::kmeans::inputCentroids;"
    PartialResults = icxx"daal::algorithms::kmeans::partialResults;"
    Centroids      = icxx"daal::algorithms::kmeans::centroids;"
    GoalFunction   = icxx"daal::algorithms::kmeans::goalFunction;"
    Assignments    = icxx"daal::algorithms::kmeans::assignments;"

    # Method enums
    const LloydDense = icxx"daal::algorithms::kmeans::lloydDense;"

    # FixMe! Might be able to avoid Vals with constant propagatio in 0.7

    Distributed(::Type{Step2Master}, ::Type{T}, nClusters::Integer, nIterations::Integer = 1) where {T<:Union{Float32,Float64}} =
        icxx"daal::algorithms::kmeans::Distributed<daal::step2Master,$T,daal::algorithms::kmeans::lloydDense>($nClusters, $nIterations);"
    Distributed(::Type{S}, nClusters, nIterations = 1) where {S} = Distributed(S, Float64, nClusters, nIterations)
    Distributed(::Type{Step1Local}, ::Type{T}, nClusters::Integer, assignFlag::Bool = false) where {T<:Union{Float32,Float64}} =
        icxx"daal::algorithms::kmeans::Distributed<daal::step1Local,$T,daal::algorithms::kmeans::lloydDense>($nClusters, $assignFlag);"

    finalizeCompute(o) = icxx"$o.finalizeCompute();"

    getPartialResult(o) = icxx"$o.getPartialResult();"

    Batch(::Type{T}, nClusters::Integer, nIterations::Integer) where {T} =
        icxx"daal::algorithms::kmeans::Batch<$T,daal::algorithms::kmeans::lloydDense>($nClusters, $nIterations);"
    Batch(nClusters::Integer, nIterations::Integer) = Batch(Float64, nClusters, nIterations)

    Base.getindex(o, ::Val{:input}) = icxx"&$o.input;"

    module Init

        import .....Daal: Step2Master, Step1Local

        using Cxx

        Data           = icxx"daal::algorithms::kmeans::init::data;"
        PartialResults = icxx"daal::algorithms::kmeans::init::partialResults;"
        Centroids      = icxx"daal::algorithms::kmeans::init::centroids;"

        # Method methods
        struct RandomDense end

        Distributed(::Type{Step2Master}, ::Type{T}, ::Type{RandomDense}, nClusters, offset = 0) where {T<:Union{Float32,Float64}} =
            icxx"daal::algorithms::kmeans::init::Distributed<daal::step2Master,$T,daal::algorithms::kmeans::init::randomDense>($nClusters, $offset);"

        Distributed(::Type{Step1Local}, ::Type{T}, ::Type{RandomDense}, nClusters, nRowsTotal, offset = 0) where {T<:Union{Float32,Float64}} =
            icxx"daal::algorithms::kmeans::init::Distributed<daal::step1Local,$T,daal::algorithms::kmeans::init::randomDense>($nClusters, $nRowsTotal, $offset);"

        get(o, id) = icxx"$o->get($id);"
    end # Init
end # Kmeans