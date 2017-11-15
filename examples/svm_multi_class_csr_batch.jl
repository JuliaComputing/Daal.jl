using Daal
using Daal.Algorithms
using Daal.Algorithms.SVM
using Daal.Algorithms.Classifier
using Daal.Algorithms.MultiClassClassifier
using Daal.Algorithms.KernelFunction
using Daal.DataManagement
using Daal.Services

include("service.jl") # e.g. pritning of results

DAAL_PREFIX = joinpath(Daal.daalrootdir, "examples", "data")

# Input data set parameters
const trainDatasetFileName     = joinpath(DAAL_PREFIX, "batch", "svm_multi_class_train_csr.csv")
const trainLabelsFileName      = joinpath(DAAL_PREFIX, "batch", "svm_multi_class_train_labels.csv")

const testDatasetFileName      = joinpath(DAAL_PREFIX, "batch", "svm_multi_class_test_csr.csv")
const testLabelsFileName       = joinpath(DAAL_PREFIX, "batch", "svm_multi_class_test_labels.csv")

const nClasses                 = 5

training = SVM.TrainingBatch()
prediction = SVM.PredictionBatch()

kernel = KernelFunction.Linear.Batch(Float64, KernelFunction.Linear.FastCSR())

function main(output = true)
    SVM.setCacheSize(training, 100000000)
    SVM.setKernel(training, kernel)
    SVM.setKernel(prediction, kernel)

    trainingResult = trainModel()

    predictionResult = testModel(trainingResult)

    printResults(predictionResult)

    return 0
end

function trainModel()
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainLabelsDataSource = DataManagement.FileDataSource(trainLabelsFileName, DataManagement.doAllocateNumericTable, DataManagement.doDictionaryFromContext)

    # Create numeric table for training data
    trainData = createSparseTable(Float64, trainDatasetFileName)

    # Retrieve the data from the input file
    DataManagement.loadDataBlock(trainLabelsDataSource)

    # Create an algorithm object to train the multi-class SVM model
    algorithm = MultiClassClassifier.TrainingBatch()

    MultiClassClassifier.setNClasses(algorithm, nClasses)
    MultiClassClassifier.setTraining(algorithm, training)
    MultiClassClassifier.setPrediction(algorithm, prediction)

    # Pass a training data set and dependent values to the algorithm
    Algorithms.setInput(algorithm, Classifier.TrainingData, trainData)
    Algorithms.setInput(algorithm, Classifier.TrainingLabels, DataManagement.getNumericTable(trainLabelsDataSource))

    # Build the multi-class SVM model
    Algorithms.compute(algorithm)

    # Retrieve the algorithm results
    return MultiClassClassifier.getResult(algorithm)
end

function testModel(trainingResult)

    # Create Numeric Tables for testing data
    testData = createSparseTable(Float64, testDatasetFileName)

    # Create an algorithm object to predict multi-class SVM values
    algorithm = MultiClassClassifier.PredictionBatch()

    MultiClassClassifier.setNClasses(algorithm, nClasses)
    MultiClassClassifier.setTraining(algorithm, training)
    MultiClassClassifier.setPrediction(algorithm, prediction)

    # Pass a testing data set and the trained model to the algorithm
    Algorithms.setInput(algorithm, Classifier.PredictionData, testData)
    Algorithms.setInput(algorithm, Classifier.PredictionModel, MultiClassClassifier.get(trainingResult, Classifier.TrainingModel))

     # Predict multi-class SVM values
    Algorithms.compute(algorithm)

    # Retrieve the algorithm results
    return MultiClassClassifier.getResult(algorithm)
end

function printResults(predictionResult)
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testLabelsDataSource = DataManagement.FileDataSource(testLabelsFileName, DataManagement.doAllocateNumericTable, DataManagement.doDictionaryFromContext)
    # Retrieve the data from input file
    DataManagement.loadDataBlock(testLabelsDataSource)
    testGroundTruth = DataManagement.getNumericTable(testLabelsDataSource)

    printNumericTables(Int32,
                       Int32,
                       testGroundTruth,
                       Classifier.get(predictionResult, Classifier.PredictionPrediction),
                       "Ground truth",
                       "Classification results",
                       "Multi-class SVM classification sample program results (first 20 observations):",
                       20)
end

main(false) # compile
@time main()
