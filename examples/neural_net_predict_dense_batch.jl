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

# TensorPtr predictionData;
predictionData = nothing
# prediction::ModelPtr predictionModel;
predictionModel = nothing
# prediction::ResultPtr predictionResult;

# void createModel();
# void testModel();
# void printResults();


# Structure that contains neural network layers identifiers
struct LayerIds
    fc1::Csize_t
    fc2::Csize_t
    sm::Csize_t
end

function configureNet()
    # Create layers of the neural network
    # Create first fully-connected layer
    # SharedPtr<fullyconnected::forward::Batch<> > fullyConnectedLayer1(new fullyconnected::forward::Batch<>(5));
    fullyConnectedLayer1 = Layers.FullyConnected.Forward.Batch(5)

    # Create second fully-connected layer
    # SharedPtr<fullyconnected::forward::Batch<> > fullyConnectedLayer2(new fullyconnected::forward::Batch<>(2));
    fullyConnectedLayer2 = Layers.FullyConnected.Forward.Batch(2)

    # Create softmax layer
    # SharedPtr<softmax::forward::Batch<> > softmaxLayer(new softmax::forward::Batch<>());
    softmaxLayer = Layers.Softmax.Forward.Batch()

    # Create topology of the neural network
    # prediction::TopologyPtr topology(new prediction::Topology());
    topology = Prediction.TopologyPtr()

    # Add layers to the topology of the neural network
    # const size_t fc1 = topology->add(fullyConnectedLayer1);
    fc1 = add(topology, fullyConnectedLayer1)
    # const size_t fc2 = topology->add(fullyConnectedLayer2);
    fc2 = add(topology, fullyConnectedLayer2)
    # const size_t sm = topology->add(softmaxLayer);
    sm = add(topology, softmaxLayer)
    # topology->get(fc1).addNext(fc2);
    NeuralNetworks.Layers.addNext(get(topology, fc1), fc2)
    # topology->get(fc2).addNext(sm);
    NeuralNetworks.Layers.addNext(get(topology, fc2), sm)
    # if(ids)
    # {
        # ids->fc1 = fc1;
        # ids->fc2 = fc2;
        # ids->sm = sm;
    # }
    ids = LayerIds(fc1, fc2, sm)
    return topology, ids
end

function main(output = true)
    createModel()

    testModel()

    printResults()

    return 0
end

function createModel()
    # Read testing data set from a .csv file and create a tensor to store input data
#     predictionData = readTensorFromCSV(testDatasetFile);
    global predictionData = readTensorFromCSV(testDatasetFile)

    # Configure the neural network
#     LayerIds ids;
#     prediction::TopologyPtr topology = configureNet(&ids);
    topology, ids = configureNet()

    # Create prediction model of the neural network
#     predictionModel = prediction::ModelPtr(new prediction::Model(*topology));
    global predictionModel = Prediction.ModelPtr(topology)

    # Read 1st fully-connected layer weights and biases from CSV file
    # 1st fully-connected layer weights are a 2D tensor of size 5 x 20
#     TensorPtr fc1Weights = readTensorFromCSV(fc1WeightsFile);
    fc1Weights = readTensorFromCSV(fc1WeightsFile)

#     # 1st fully-connected layer biases are a 1D tensor of size 5
# #     TensorPtr fc1Biases = readTensorFromCSV(fc1BiasesFile);
    fc1Biases = readTensorFromCSV(fc1BiasesFile)

    # Set weights and biases of the 1st fully-connected layer
#     forward::Input *fc1Input = predictionModel->getLayer(ids.fc1)->getLayerInput();

    fc1Input = Layers.getLayerInput(Prediction.getLayer(predictionModel, ids.fc1))
#     fc1Input->set(forward::weights, fc1Weights);
    set(fc1Input, Layers.Forward.Weights, fc1Weights)
#     fc1Input->set(forward::biases, fc1Biases);
    set(fc1Input, Layers.Forward.Biases, fc1Biases)

    # Set flag that specifies that weights and biases of the 1st fully-connected layer are initialized
#     predictionModel->getLayer(ids.fc1)->getLayerParameter()->weightsAndBiasesInitialized = true;
    Layers.getLayerParameter(Prediction.getLayer(predictionModel, ids.fc1))[Val{:weightsAndBiasesInitialized}()] = true

    # Read 2nd fully-connected layer weights and biases from CSV file
    # 2nd fully-connected layer weights are a 2D tensor of size 2 x 5
    # TensorPtr fc2Weights = readTensorFromCSV(fc2WeightsFile);
    fc2Weights = readTensorFromCSV(fc2WeightsFile)

#     /* 2nd fully-connected layer biases are a 1D tensor of size 2 */
#     TensorPtr fc2Biases = readTensorFromCSV(fc2BiasesFile);
    fc2Biases = readTensorFromCSV(fc2BiasesFile)

#     # Set weights and biases of the 2nd fully-connected layer
#     forward::Input *fc2Input = predictionModel->getLayer(ids.fc2)->getLayerInput();
    fc2Input = Layers.getLayerInput(Prediction.getLayer(predictionModel, ids.fc2))
#     fc2Input->set(forward::weights, fc2Weights);
    set(fc2Input, Layers.Forward.Weights, fc2Weights)
#     fc2Input->set(forward::biases, fc2Biases);
    set(fc2Input, Layers.Forward.Biases, fc2Biases)

    # Set flag that specifies that weights and biases of the 2nd fully-connected layer are initialized
    # predictionModel->getLayer(ids.fc2)->getLayerParameter()->weightsAndBiasesInitialized = true;
    Layers.getLayerParameter(Prediction.getLayer(predictionModel, ids.fc2))[Val{:weightsAndBiasesInitialized}()] = true
end

function testModel()
    # Create an algorithm to compute the neural network predictions
#     prediction::Batch<> net;
    net = Prediction.Batch()

    # Set parameters for the prediction neural network
#     net.parameter.batchSize = predictionData->getDimensionSize(0);
    net[Val{:parameter}()][Val{:batchSize}()] = getDimensionSize(predictionData, 0)

    # Set input objects for the prediction neural network
#     net.input.set(prediction::model, predictionModel);
    set(net[Val{:input}()], Prediction.ModelId, predictionModel)
#     net.input.set(prediction::data, predictionData);
    set(net[Val{:input}()], Prediction.DataId, predictionData)

    # Run the neural network prediction
#     net.compute();
    compute(net)

    # Print results of the neural network prediction
#     predictionResult = net.getResult();
    global predictionResult = Prediction.getResult(net)
end

function printResults()
    # Read testing ground truth from a .csv file and create a tensor to store the data
#     TensorPtr predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);
    predictionGroundTruth = readTensorFromCSV(testGroundTruthFile)

#     printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
#                              "Ground truth", "Neural network predictions: each class probability",
#                              "Neural network classification results (first 20 observations):", 20);
    printTensors(Int32, Float32,
        predictionGroundTruth,
        get(predictionResult, Prediction.PredictionId),
        "Ground truth",
        "Neural network predictions: each class probability",
        "Neural network classification results (first 20 observations):",
        20)
end

main()
# @time main()
