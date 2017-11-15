module Algorithms

using Cxx

compute(o) = icxx"$(o.o).compute();"

setInput(o, id, value) = icxx"$(o.o).input.set($id, $(value.o));"
addInput(o, id, value) = icxx"$(o.o).input.add($id, $(value.o));"

include("pca.jl")
include("svm.jl")
include("kernelfunction.jl")
include("classifier.jl")
include("multiclassclassifier.jl")
include("kmeans.jl")

end