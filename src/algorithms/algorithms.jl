module Algorithms

using Cxx
using Cxx: CppValue

const CppPtrs = Union{Cxx.CppPtr,
                      Cxx.CppRef,
                      Cxx.CppValue{T,N} where {T<:Cxx.CxxQualType{S,CVR},N} where {S<:Cxx.CppTemplate{Cxx.CppBaseType{Symbol("daal::services::interface1::SharedPtr")},targs},CVR} where {targs}
                      }

import Base: get
export add, compute, getResult, set

# compute(o) = icxx"$(o.o).compute();"
compute(o) = icxx"$o.compute();"

get(o::CppPtrs, id) = icxx"$o->get($id);"
# set(o, id, value) = icxx"$(o.o)->set($id, $(value.o));"
set(o::CppPtrs, id, value) = icxx"$o->set($id, $value);"
# add(o, id, value) = icxx"$(o.o)->add($id, $(value.o));"
add(o::CppPtrs, value)     = icxx"$o->add($value);"
add(o::CppPtrs, id, value) = icxx"$o->add($id, $value);"

getResult(o::Cxx.CppValue) = icxx"$o.getResult();"

Base.getindex(o::CppValue, ::Val{:input})     = icxx"&$o.input;"
Base.getindex(o::CppPtrs,  ::Val{:input})     = icxx"&$o->input;"
Base.getindex(o::CppValue, ::Val{:parameter}) = icxx"&$o.parameter;"
Base.getindex(o::CppPtrs,  ::Val{:parameter}) = icxx"&$o->parameter;"

Base.setindex!(o::CppValue, value, ::Val{:kernel})    = icxx"$o.kernel = $value;"
Base.setindex!(o::CppPtrs,  value, ::Val{:kernel})    = icxx"$o->kernel = $value;"
Base.setindex!(o::CppValue, value, ::Val{:cacheSize}) = icxx"$o.cacheSize = $value;"
Base.setindex!(o::CppPtrs,  value, ::Val{:cacheSize}) = icxx"$o->cacheSize = $value;"
Base.setindex!(o::CppValue, value, ::Val{:batchSize}) = icxx"$o.batchSize = $value;"
Base.setindex!(o::CppPtrs,  value, ::Val{:batchSize}) = icxx"$o->batchSize = $value;"
Base.setindex!(o::CppValue, value, ::Val{:weightsAndBiasesInitialized}) =
    icxx"$o.weightsAndBiasesInitialized = $value;"
Base.setindex!(o::CppPtrs, value, ::Val{:weightsAndBiasesInitialized}) =
    icxx"$o->weightsAndBiasesInitialized = $value;"

include("pca.jl")
include("svm.jl")
include("kernelfunction.jl")
include("classifier.jl")
include("multiclassclassifier.jl")
include("kmeans.jl")
include("neural_networks.jl")

end