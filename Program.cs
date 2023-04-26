using Microsoft.ML;
using Microsoft.ML.Data;

using TransferLearningTF;

const string input = "input";
const string softmax2 = "softmax2_pre_activation";
const string labelKey = "LabelKey";
const string predictedLabel = "PredictedLabel";

string _assetsPath = Path.Combine(Directory.GetParent(Directory.GetParent(Directory.GetParent(Environment.CurrentDirectory)!.FullName)!.FullName)!.FullName, "assets");
string _imagesFolder = Path.Combine(_assetsPath, "images");
string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

var context = new MLContext();

ITransformer model = GenerateModel(context);

ClassifySingleImage(context, model);

Console.ReadKey();

ITransformer GenerateModel(MLContext context)
{
    IEstimator<ITransformer> pipeline = context.Transforms.LoadImages(input, _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
        .Append(context.Transforms.ResizeImages(input, InceptionSettings.ImageWidth, InceptionSettings.ImageHeight, inputColumnName: input))
        .Append(context.Transforms.ExtractPixels(input, interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
        .Append(context.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).ScoreTensorFlowModel(new[] { softmax2 }, new[] { input }, true))
        .Append(context.Transforms.Conversion.MapValueToKey(labelKey, inputColumnName: nameof(ImageData.Label)))
        .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: labelKey, featureColumnName: softmax2))
        .Append(context.Transforms.Conversion.MapKeyToValue(nameof(ImagePrediction.PredictedLabelValue), predictedLabel))
        .AppendCacheCheckpoint(context);

    IDataView trainingData = context.Data.LoadFromTextFile<ImageData>(_trainTagsTsv, hasHeader: false);

    ITransformer model = pipeline.Fit(trainingData);

    IDataView predictions = model.Transform(context.Data.LoadFromTextFile<ImageData>(_testTagsTsv, hasHeader: false));

    DisplayResults(context.Data.CreateEnumerable<ImagePrediction>(predictions, true));

    MulticlassClassificationMetrics metrics
        = context.MulticlassClassification.Evaluate(predictions, labelColumnName: labelKey, predictedLabelColumnName: predictedLabel);

    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {string.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

    return model;
}
void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (var prediction in imagePredictionData)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
    }
}
void ClassifySingleImage(MLContext context, ITransformer model)
{
    var imageData = new ImageData
    {
        ImagePath = _predictSingleImage
    };
    var predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    var prediction = predictor.Predict(imageData);

    Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
}