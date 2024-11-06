using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

class Program
{
    // Structure for data from CSV
    public class AnomalyData
    {
        [LoadColumn(0)] // Load the DateTime column
        public string DateTime { get; set; }

        [LoadColumn(1)] // Load the Accelerometer1RMS column
        public float Accelerometer1RMS { get; set; }

        [LoadColumn(2)] // Load the Accelerometer2RMS column
        public float Accelerometer2RMS { get; set; }

        [LoadColumn(3)] // Load the Current column
        public float Current { get; set; }

        [LoadColumn(4)] // Load the Pressure column
        public float Pressure { get; set; }

        [LoadColumn(5)] // Load the Temperature column
        public float Temperature { get; set; }

        [LoadColumn(6)] // Load the Thermocouple column
        public float Thermocouple { get; set; }

        [LoadColumn(7)] // Load the Voltage column
        public float Voltage { get; set; }

        [LoadColumn(8)] // Load the Volume Flow Rate RMS column
        public float VolumeFlowRateRMS { get; set; }

        [LoadColumn(9)] // Load the Anomaly column
        public float Anomaly { get; set; }

        [LoadColumn(10)] // Load the Changepoint column
        public float Changepoint { get; set; }
    }

    public class AnomalyPrediction
    {
        public float Score { get; set; }
    }

    static void Main(string[] args)
    {
        // Create MLContext object (entry point to ML.NET)
        var context = new MLContext();

        // Define paths to the data files and directories
        string anomalyFreeDataPath = @"C:\Users\stere\source\repos\Lab4\archive\SKAB\anomaly-free\anomaly-free.csv";
        string valve1DataPath = @"C:\Users\stere\source\repos\Lab4\archive\SKAB\valve1";
        string valve2DataPath = @"C:\Users\stere\source\repos\Lab4\archive\SKAB\valve2";
        string otherDataPath = @"C:\Users\stere\source\repos\Lab4\archive\SKAB\other";

        // Load the CSV data from various sources
        var anomalyFreeData = LoadCsvData(context, anomalyFreeDataPath);
        var valve1Data = LoadCsvDataFromFolder(context, valve1DataPath);
        var valve2Data = LoadCsvDataFromFolder(context, valve2DataPath);
        var otherData = LoadCsvDataFromFolder(context, otherDataPath);

        // Combine all datasets into one list
        var fullData = anomalyFreeData.Concat(valve1Data).Concat(valve2Data).Concat(otherData).ToList();

        // Split the data into training and testing datasets (80% for training, 20% for testing)
        var trainTestSplit = context.Data.TrainTestSplit(context.Data.LoadFromEnumerable(fullData), testFraction: 0.2);

        // Create the data processing pipeline
        var pipeline = context.Transforms.Concatenate("Features", nameof(AnomalyData.Accelerometer1RMS),
                                                       nameof(AnomalyData.Accelerometer2RMS),
                                                       nameof(AnomalyData.Current),
                                                       nameof(AnomalyData.Pressure),
                                                       nameof(AnomalyData.Temperature),
                                                       nameof(AnomalyData.Thermocouple),
                                                       nameof(AnomalyData.Voltage),
                                                       nameof(AnomalyData.VolumeFlowRateRMS))
                        .Append(context.Transforms.NormalizeMeanVariance("Features"))
                        .Append(context.AnomalyDetection.Trainers.RandomizedPca(featureColumnName: "Features", rank: 1));

        // Train the model on the training dataset
        var model = pipeline.Fit(trainTestSplit.TrainSet);

        // Save the trained model to a file
        string modelPath = @"C:\Users\stere\source\repos\Lab4\model.zip";
        context.Model.Save(model, trainTestSplit.TrainSet.Schema, modelPath);
        Console.WriteLine($"Model saved to {modelPath}");

        // Load the model from the saved file
        ITransformer loadedModel = context.Model.Load(modelPath, out var modelInputSchema);
        Console.WriteLine("Model loaded from file");

        // Predict anomalies on the test dataset using the loaded model
        var predictions = loadedModel.Transform(trainTestSplit.TestSet);

        // Output anomaly scores for the predictions
        var predictionResults = context.Data.CreateEnumerable<AnomalyPrediction>(predictions, reuseRowObject: false).ToList();
        foreach (var result in predictionResults)
        {
            Console.WriteLine($"Anomaly Score: {result.Score}");
        }

        // Make a prediction for a new sample data point using the loaded model
        var sampleData = new AnomalyData
        {
            DateTime = "10.03.2020 14:00",
            Accelerometer1RMS = 0.28f,
            Accelerometer2RMS = 0.30f,
            Current = 1.75f,
            Pressure = 0.45f,
            Temperature = 72.3f,
            Thermocouple = 26.5f,
            Voltage = 230f,
            VolumeFlowRateRMS = 120f
        };

        // Create a prediction engine and predict the anomaly score for the sample data
        var predictionEngine = context.Model.CreatePredictionEngine<AnomalyData, AnomalyPrediction>(loadedModel);
        var prediction = predictionEngine.Predict(sampleData);
        Console.WriteLine($"Anomaly Score: {prediction.Score}");

        // Wait for user input before exiting
        Console.WriteLine("Press Enter to exit...");
        Console.ReadLine();
    }

    // Load CSV data from a single file with filtering for non-empty data
    static List<AnomalyData> LoadCsvData(MLContext context, string filePath)
    {
        // Load data from the file using the LoadFromTextFile method
        var data = context.Data.LoadFromTextFile<AnomalyData>(filePath, separatorChar: ';', hasHeader: true);

        // Convert the IDataView into a list of AnomalyData objects
        var dataList = context.Data.CreateEnumerable<AnomalyData>(data, reuseRowObject: false)
            .Where(d => {
                return !string.IsNullOrWhiteSpace(d.DateTime) &&
                       !float.IsNaN(d.Accelerometer1RMS) && !float.IsInfinity(d.Accelerometer1RMS) &&
                       !float.IsNaN(d.Accelerometer2RMS) && !float.IsInfinity(d.Accelerometer2RMS) &&
                       !float.IsNaN(d.Current) && !float.IsInfinity(d.Current) &&
                       !float.IsNaN(d.Pressure) && !float.IsInfinity(d.Pressure) &&
                       !float.IsNaN(d.Temperature) && !float.IsInfinity(d.Temperature) &&
                       !float.IsNaN(d.Thermocouple) && !float.IsInfinity(d.Thermocouple) &&
                       !float.IsNaN(d.Voltage) && !float.IsInfinity(d.Voltage) &&
                       !float.IsNaN(d.VolumeFlowRateRMS) && !float.IsInfinity(d.VolumeFlowRateRMS);
            })
            .ToList();

        return dataList;
    }

    // Load CSV data from all files in a folder with filtering for non-empty data
    static List<AnomalyData> LoadCsvDataFromFolder(MLContext context, string folderPath)
    {
        // Get all CSV files from the folder
        var files = Directory.GetFiles(folderPath, "*.csv");
        var allData = new List<AnomalyData>();

        // Loop through all files and load data from each with filtering
        foreach (var file in files)
        {
            var data = LoadCsvData(context, file);
            allData.AddRange(data); // Add data from each file to the list
        }

        return allData;
    }
}
