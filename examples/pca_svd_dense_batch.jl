using Daal
using Daal.Algorithms
using Daal.Algorithms.PCA
using Daal.DataManagement

include("service.jl") # e.g. pritning of results

DAAL_PREFIX = joinpath(Daal.daalrootdir, "examples", "data")

# Input data set parameters
dataFileName = joinpath(DAAL_PREFIX, "batch", "pca_normalized.csv")
nVectors = 1000

function main(output = true)
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = DataManagement.FileDataSource(dataFileName, DataManagement.doAllocateNumericTable, DataManagement.doDictionaryFromContext)

    # Retrieve the data from the input file
    DataManagement.loadDataBlock(dataSource, nVectors)

    # Create an algorithm for principal component analysis using the SVD method
    algorithm = PCA.Batch(Float32, svd)

    # Set the algorithm input data
    PCA.setInput(algorithm, PCA.Data, DataManagement.getNumericTable(dataSource))

    # Compute results of the PCA algorithm
    Algorithms.compute(algorithm)
    result = PCA.getResult(algorithm)

    # Print the results
    printNumericTable(PCA.get(result, PCA.Eigenvalues), "Eigenvalues:")
    printNumericTable(PCA.get(result, PCA.Eigenvectors), "Eigenvectors:")

    return 0
end

main(false) # compile
@time main()
