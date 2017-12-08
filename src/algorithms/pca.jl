module PCA

using Cxx

const Data         = icxx"daal::algorithms::pca::data;"
const Eigenvalues  = icxx"daal::algorithms::pca::eigenvalues;"
const Eigenvectors = icxx"daal::algorithms::pca::eigenvectors;"

Batch(::Type{T}, ::typeof(svd)) where {T<:Union{Float32,Float64}} = icxx"daal::algorithms::pca::Batch<$T,daal::algorithms::pca::svdDense>();"

end