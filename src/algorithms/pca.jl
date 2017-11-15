module PCA

import ....Daal.DataManagement: NumericTable

using Cxx

const Data         = icxx"daal::algorithms::pca::data;"
const Eigenvalues  = icxx"daal::algorithms::pca::eigenvalues;"
const Eigenvectors = icxx"daal::algorithms::pca::eigenvectors;"

struct Batch{T}
    o::Cxx.CppValue
end

Batch(::Type{T}, ::typeof(svd)) where {T<:Union{Float32,Float64}} = Batch{T}(icxx"daal::algorithms::pca::Batch<$T,daal::algorithms::pca::svdDense>();")

setInput(o::Batch, id, value) = icxx"$(o.o).input.set($id, $(value.o));"

struct Result
    o::Cxx.CppValue
end

getResult(batch::Batch) = Result(icxx"daal::services::SharedPtr<daal::algorithms::pca::Result> result = $(batch.o).getResult(); result;")

get(o::Result, what) = NumericTable(icxx"daal::services::SharedPtr<daal::data_management::NumericTable> result = $(o.o)->get($what); result;")

end