using Daal
using Daal.Algorithms
using Daal.Algorithms.NeuralNetworks
using Daal.Algorithms.NeuralNetworks.Prediction
using Daal.Algorithms.NeuralNetworks.Layers

include("service.jl") # e.g. pritning of results

DAAL_PREFIX = joinpath(Daal.daalrootdir, "examples", "data")

# Input data set parameters
testDatasetFile     = joinpath(DAAL_PREFIX, "batch", "neural_network_test.csv")
testGroundTruthFile = joinpath(DAAL_PREFIX, "batch", "neural_network_test_ground_truth.csv")

# Weights and biases obtained on the training stage
fc1WeightsFile = joinpath(DAAL_PREFIX, "batch", "fc1_weights.csv")
fc1BiasesFile  = joinpath(DAAL_PREFIX, "batch", "fc1_biases.csv")
fc2WeightsFile = joinpath(DAAL_PREFIX, "batch", "fc2_weights.csv")
fc2BiasesFile  = joinpath(DAAL_PREFIX, "batch", "fc2_biases.csv")

# Structure that contains neural network layers identifiers
struct LayerIds
    fc1::Csize_t
    fc2::Csize_t
    sm::Csize_t
end

function configureNet()
    # Create layers of the neural network
    # Create first fully-connected layer
    fullyConnectedLayer1 = Layers.FullyConnected.Forward.Batch(5)

    # Create second fully-connected layer
    fullyConnectedLayer2 = Layers.FullyConnected.Forward.Batch(2)

    # Create softmax layer
    softmaxLayer = Layers.Softmax.Forward.Batch()

    # Create topology of the neural network
    topology = Prediction.TopologyPtr()

    # Add layers to the topology of the neural network
    fc1 = add(topology, fullyConnectedLayer1)
    fc2 = add(topology, fullyConnectedLayer2)
    sm = add(topology, softmaxLayer)
    NeuralNetworks.Layers.addNext(get(topology, fc1), fc2)
    NeuralNetworks.Layers.addNext(get(topology, fc2), sm)
    ids = LayerIds(fc1, fc2, sm)

    return topology, ids
end

function main(output = true)
    createModel()

    testModel()

    output && printResults()

    return 0
end

function createModel()
    # Read testing data set from a .csv file and create a tensor to store input data
    global predictionData = readTensorFromCSV(testDatasetFile)

    # Configure the neural network
    topology, ids = configureNet()

    # Create prediction model of the neural network
    global predictionModel = Prediction.ModelPtr(topology)

    # Read 1st fully-connected layer weights and biases from CSV file
    # 1st fully-connected layer weights are a 2D tensor of size 5 x 20
    fc1Weights = readTensorFromCSV(fc1WeightsFile)

    # 1st fully-connected layer biases are a 1D tensor of size 5
    fc1Biases = readTensorFromCSV(fc1BiasesFile)

    # Set weights and biases of the 1st fully-connected layer
    fc1Input = Layers.getLayerInput(Prediction.getLayer(predictionModel, ids.fc1))
    set(fc1Input, Layers.Forward.Weights, fc1Weights)
    set(fc1Input, Layers.Forward.Biases, fc1Biases)

    # Set flag that specifies that weights and biases of the 1st fully-connected layer are initialized
    Layers.getLayerParameter(Prediction.getLayer(predictionModel, ids.fc1))[Val{:weightsAndBiasesInitialized}()] = true

    # Read 2nd fully-connected layer weights and biases from CSV file
    # 2nd fully-connected layer weights are a 2D tensor of size 2 x 5
    fc2Weights = readTensorFromCSV(fc2WeightsFile)

    # 2nd fully-connected layer biases are a 1D tensor of size 2
    fc2Biases = readTensorFromCSV(fc2BiasesFile)

    # Set weights and biases of the 2nd fully-connected layer
    fc2Input = Layers.getLayerInput(Prediction.getLayer(predictionModel, ids.fc2))
    set(fc2Input, Layers.Forward.Weights, fc2Weights)
    set(fc2Input, Layers.Forward.Biases, fc2Biases)

    # Set flag that specifies that weights and biases of the 2nd fully-connected layer are initialized
    Layers.getLayerParameter(Prediction.getLayer(predictionModel, ids.fc2))[Val{:weightsAndBiasesInitialized}()] = true
end

function testModel()
    # Create an algorithm to compute the neural network predictions
    net = Prediction.Batch()

    # Set parameters for the prediction neural network
    net[Val{:parameter}()][Val{:batchSize}()] = getDimensionSize(predictionData, 0)

    # Set input objects for the prediction neural network
    set(net[Val{:input}()], Prediction.ModelId, predictionModel)
    set(net[Val{:input}()], Prediction.DataId, predictionData)

    # Run the neural network prediction
    compute(net)

    # Print results of the neural network prediction
    global predictionResult = getResult(net)
end

function printResults()
    # Read testing ground truth from a .csv file and create a tensor to store the data
    predictionGroundTruth = readTensorFromCSV(testGroundTruthFile)

    printTensors(Int32, Float32,
        predictionGroundTruth,
        get(predictionResult, Prediction.PredictionId),
        "Ground truth",
        "Neural network predictions: each class probability",
        "Neural network classification results (first 20 observations):",
        20)
end

main(false)
@time main()
