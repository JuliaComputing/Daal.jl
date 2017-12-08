module Algorithms

using Cxx

const CppType = Union{Cxx.CppValue,Cxx.CppPtr,Cxx.CppRef}

import Base: get
export add, compute, getResult, set

# compute(o) = icxx"$(o.o).compute();"
compute(o) = icxx"$o.compute();"

get(o::CppType, id) = icxx"$o->get($id);"
# set(o, id, value) = icxx"$(o.o)->set($id, $(value.o));"
set(o::CppType, id, value) = icxx"$o->set($id, $value);"
# add(o, id, value) = icxx"$(o.o)->add($id, $(value.o));"
add(o::CppType, id, value) = icxx"$o->add($id, $value);"

getResult(o::CppType) = icxx"$o.getResult();"

struct Parameter
    o::Cxx.CppPtr
end

# Should eventually become a generated function
for p in (:batchSize, :weightsAndBiasesInitialized)
    @eval begin
        qp = $(QuoteNode(p))
        Base.setindex!(o::Parameter, value::Number, ::Val{qp}) = @icxx_str $"\$(o.o)->$p = \$value;"
        Base.setindex!(o::Parameter, value        , ::Val{qp}) = @icxx_str $"\$(o.o)->$p = \$(value.o);"
    end
end
# @generated function Base.setindex!(o::Parameter, value, ::Val{K}) where {K}
#     ccall(:jl_, Void, (Any,), :("$(o).o->$K"))
#     if value <: Number
#         # s = icxx"$(o).o->$K = $value"
#         return :(icxx"$(o.o)->$(K) = $value;")
#     else
#         # s = icxx"$(o).o->$K = $(value).o"
#         return :(icxx"$(o.o)->$(K) = $(value.o);")
#     end
#     # return :($s)
# end

include("pca.jl")
include("svm.jl")
include("kernelfunction.jl")
include("classifier.jl")
include("multiclassclassifier.jl")
include("kmeans.jl")
include("neural_networks.jl")

end