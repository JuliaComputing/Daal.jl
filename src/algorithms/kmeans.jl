module Kmeans

import ....Daal: Step2Master, Step1Local
import ....Daal.DataManagement: NumericTable

using Cxx

Data           = icxx"daal::algorithms::kmeans::data;"
InputCentroids = icxx"daal::algorithms::kmeans::inputCentroids;"
PartialResults = icxx"daal::algorithms::kmeans::partialResults;"
Centroids      = icxx"daal::algorithms::kmeans::centroids;"
GoalFunction   = icxx"daal::algorithms::kmeans::goalFunction;"
Assignments    = icxx"daal::algorithms::kmeans::assignments;"

# Method methods
struct LloydDense end

struct Distributed{S,T,M}
    o
end

Distributed(::Type{Step2Master}, ::Type{T}, nClusters, nIterations = 1) where {T<:Union{Float32,Float64}} =
    Distributed{Step2Master,T,LloydDense}(icxx"daal::algorithms::kmeans::Distributed<daal::step2Master,$T,daal::algorithms::kmeans::lloydDense>($nClusters, $nIterations);")
Distributed(::Type{S}, nClusters, nIterations = 1) where {S} = Distributed(S, Float64, nClusters, nIterations)

Distributed(::Type{Step1Local}, ::Type{T}, nClusters, assignFlag = false) where {T<:Union{Float32,Float64}} =
    Distributed{Step1Local,T,LloydDense}(icxx"daal::algorithms::kmeans::Distributed<daal::step1Local,$T,daal::algorithms::kmeans::lloydDense>($nClusters, $assignFlag);")

finalizeCompute(o::Distributed) = icxx"$(o.o).finalizeCompute();"

struct PartialResult
    o::Cxx.CppValue
end
getPartialResult(o::Distributed) = PartialResult(icxx"$(o.o).getPartialResult();")

struct Result
    o::Cxx.CppValue
end
getResult(o::Distributed) = Result(icxx"$(o.o).getResult();")

get(o::Result, id) = NumericTable(icxx"$(o.o)->get($id);")

struct Batch{T<:Union{Float32,Float64},M}
    o::Cxx.CppValue
end
Batch(::Type{T}, nClusters::Integer, nIterations::Integer) where {T} =
    Batch{T,LloydDense}(icxx"daal::algorithms::kmeans::Batch<$T,daal::algorithms::kmeans::lloydDense>($nClusters, $nIterations);")
Batch(nClusters::Integer, nIterations::Integer) = Batch(Float64, nClusters, nIterations)

getResult(o::Batch) = Result(icxx"""
    daal::services::SharedPtr<daal::algorithms::kmeans::Result> result = $(o.o).getResult();
    result;
    """)

module Init

import .....Daal: Step2Master, Step1Local
import .....Daal.DataManagement: NumericTable

using Cxx

Data           = icxx"daal::algorithms::kmeans::init::data;"
PartialResults = icxx"daal::algorithms::kmeans::init::partialResults;"
Centroids      = icxx"daal::algorithms::kmeans::init::centroids;"

# Method methods
struct RandomDense end

struct Distributed{S,T,M}
    o::Cxx.CppValue
end

Distributed(::Type{Step2Master}, ::Type{T}, ::Type{RandomDense}, nClusters, offset = 0) where {T<:Union{Float32,Float64}} =
    Distributed{Step2Master,T,RandomDense}(icxx"daal::algorithms::kmeans::init::Distributed<daal::step2Master,$T,daal::algorithms::kmeans::init::randomDense>($nClusters, $offset);")

Distributed(::Type{Step1Local}, ::Type{T}, ::Type{RandomDense}, nClusters, nRowsTotal, offset = 0) where {T<:Union{Float32,Float64}} =
    Distributed{Step1Local,T,RandomDense}(icxx"daal::algorithms::kmeans::init::Distributed<daal::step1Local,$T,daal::algorithms::kmeans::init::randomDense>($nClusters, $nRowsTotal, $offset);")

finalizeCompute(o::Distributed) = icxx"$(o.o).finalizeCompute();"

struct PartialResult
    o::Cxx.CppValue
end
getPartialResult(o::Distributed) = PartialResult(icxx"$(o.o).getPartialResult();")

struct Result
    o::Cxx.CppValue
end
getResult(o::Distributed) = Result(icxx"$(o.o).getResult();")

get(o::Result, id) = NumericTable(icxx"$(o.o)->get($id);")

end

end