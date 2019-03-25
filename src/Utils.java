import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RenameAttribute;

public class Utils {

    public static void  runNNWithClustersAsOnlyFeatures(String dataFile, float l, float m, int n) throws Exception {
        String dataset = "winequalitywhite";
        // transform data to only have these features:
        // 1. Clusters from KMeans
        // 2. Clusters from EM

        Instances data = readDataFile("datasets/" + dataset + "/" + dataFile + ".arff");
        // add kmeans cluster
        AddCluster addKM = new AddCluster();
        addKM.setClusterer(new KMeans().getClusterer(3));
        addKM.setInputFormat(data);
        data = AddCluster.useFilter(data, addKM);

//        System.out.println("Added kmeans cluster");

        // rename cluster
        RenameAttribute renameAttribute = new RenameAttribute();
        renameAttribute.setFind("cluster");
        renameAttribute.setReplace("kmeansCluster");
        renameAttribute.setInputFormat(data);
        data = RenameAttribute.useFilter(data, renameAttribute);

//        System.out.println("Renamed kmeans cluster");

        // add em cluster
        AddCluster addEM = new AddCluster();
        addEM.setClusterer(new ExpectationMaximization().getClusterer(13));
        addEM.setIgnoredAttributeIndices(data.numAttributes() + "");
        addEM.setInputFormat(data);
        data = AddCluster.useFilter(data, addEM);

//        System.out.println("Added EM cluster");

        // remove original attributes
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(new int[] {data.numAttributes()-3, data.numAttributes()-2, data.numAttributes()-1});
        removeFilter.setInvertSelection(true);
        removeFilter.setInputFormat(data);
        data = Filter.useFilter(data, removeFilter);

//        System.out.println("Removed attributes");
        System.out.println("Num attributes: " + data.numAttributes());
        System.out.println("Class index is " + data.classIndex());
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.print((i+1) + ":" + data.attribute(i) + "\t");
        }
//        System.out.println("Running NN");

//        new NeuralNetwork(.1f, .7f, 700).findScores(data);
    }

    public static Instances[] getSplits(Instances data, int numSplits) throws Exception {
        Instances result = null;
        Instances[] splits = new Instances[numSplits];

        for (int i = 1; i <= numSplits; i++) {
            StratifiedRemoveFolds strRmvFolds = new StratifiedRemoveFolds();
            strRmvFolds.setFold(i);
            strRmvFolds.setNumFolds(numSplits);
            strRmvFolds.setSeed(0);
            strRmvFolds.setInvertSelection(false);
            strRmvFolds.setInputFormat(data);
            Instances addFold = StratifiedRemoveFolds.useFilter(data, strRmvFolds);
            if (result == null) {
                result = addFold;
            } else {
                result = mergeInstances(result, addFold);
            }
            splits[i-1] = new Instances(result);
        }

        return splits;
    }

    // taken from https://stackoverflow.com/questions/10771558/how-to-merge-two-sets-of-weka-instances-together
    public static Instances mergeInstances(Instances data1, Instances data2)
            throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                    (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                            .setValue(i,instance.stringValue(i));
                }
            }
        }

        return dest;
    }

    public static Instances readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }
        Instances data = null;
        try {
            data = new Instances(inputReader);
        } catch (IOException e) {
            e.printStackTrace();
        }

        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    public static Instances getWithoutClass(Instances data) throws Exception {
        Remove filter = new Remove();
        filter.setAttributeIndices("" + data.numAttributes());
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    public static Instances keepFirstNAttributes(Instances data, int n, boolean keepClass) throws Exception {
        Remove filter = new Remove();
        String start = "" + (n+1);
        String end = "last";
        if (keepClass) {
            end = "" + (data.numAttributes() - 1);
        }
        filter.setAttributeIndices(start + "-" + end);
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);

        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);

        return evaluation;
    }

    public static double calculateAccuracy(Classifier model, ArrayList<Prediction> predictions, String fileName) throws IOException {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.get(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        double accuracy = 100 * correct / predictions.size();

        if (fileName == null || fileName.equals("")) {
            return accuracy;
        }

        // Print current classifier's name and accuracy in a complicated,
        // but nice-looking way.
//        System.out.println("Accuracy of " + model.getClass().getSimpleName() + ": " + String.format("%.2f%%", accuracy)
//                + "\n---------------------------------");

        // write to file in results folder
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        writer.write(model.toString());
        writer.close();

        // Uncomment to see the summary for each training-testing pair.
//        System.out.println(model.toString());
        return accuracy;
    }

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }

    public static double TrainTest(Classifier model, Instances train, String fileName) throws Exception {
//        System.out.println("Train results:");
        // Collect every group of predictions for current model
        ArrayList<Prediction> predictions = new ArrayList<Prediction>();

        Evaluation validation = classify(model, train, train);

        predictions.addAll(validation.predictions());

        // Uncomment to see the summary for each training-testing pair.
        // System.out.println(model.toString());

        // Calculate overall accuracy of current classifier on all splits
        return calculateAccuracy(model, predictions, fileName);
    }

    public static double TestTest(Classifier model, Instances train, Instances test, String fileName) throws Exception {
//        System.out.println("Test results:");
        // Collect every group of predictions for current model
        ArrayList<Prediction> predictions = new ArrayList<Prediction>();

        Evaluation validation = classify(model, train, test);

        predictions.addAll(validation.predictions());

        // Uncomment to see the summary for each training-testing pair.
        // System.out.println(model.toString());

        // Calculate overall accuracy of current classifier on all splits
        return calculateAccuracy(model, predictions, fileName);
    }

    public static double CVTest(Classifier model, Instances train, String fileName) throws Exception {
//        if (fileName != null && !fileName.equals("")) {
//            System.out.println("CV-5 results:");
//        }
        // Collect every group of predictions for current model
        ArrayList<Prediction> predictions = new ArrayList<Prediction>();

        // Do 5-fold cross validation split
        Instances[][] split = crossValidationSplit(train, 5);

        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];

        // For each training-testing split pair, train and test the classifier
        for (int i = 0; i < trainingSplits.length; i++) {
            Evaluation validation = classify(model, trainingSplits[i], testingSplits[i]);

            predictions.addAll(validation.predictions());
        }

        // Calculate overall accuracy of current classifier on all splits
        return calculateAccuracy(model, predictions, fileName);
    }

    public static void runTests(Classifier model, int split, Instances dataSplit, StringBuilder result, ArrayList<Double> cvScores) throws Exception {
//        Instances train = readDataFile("datasets/" + dataset + "/splits/" + split + ".arff");

        // Commented out trainScore and testScore, so only using cvScore
//        trainScores.add(TrainTest(model, train, dir + split + " train"));

        long startTime = 0;
        if (split == 100) {
            startTime = System.currentTimeMillis();
        }
//        testScores.add(TestTest(model, train, test, dir + split + " test"));
//        if (split == 100) {
//            String timeResult = "Time to run: " + ((System.currentTimeMillis() - startTime) / 1000f) + " seconds" + "\n";
//            result.append(timeResult);
//        }

//        Instances train = readDataFile("datasets/" + dataset + "/" + dataFile + ".arff");
        cvScores.add(CVTest(model, dataSplit, null));

        if (split == 100) {
            float timeToRun = ((System.currentTimeMillis() - startTime) / 1000f / 5);
//            String timeResult = "Time to run: " + timeToRun + " seconds" + "\n";
//            result.append(timeResult);
            result.append(timeToRun);
        }
    }

    public static String getScoresString(ArrayList<Double> scores) {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        for (int i = 0; i < scores.size(); i++) {
            double score = scores.get(i);
            builder.append(score + ((i == scores.size() - 1) ? "" : ", "));
        }
        builder.append("]");
        return builder.toString();
    }

    public static String getDir(String dataset, String modelName) {
        return "results/" + dataset + "/" + modelName + "/";
    }

    public static void findScores(Classifier model, String dataset, String dataFile, String modelName) throws Exception {
        Instances data = Utils.readDataFile("datasets/" + dataset + "/" + dataFile + ".arff");
        findScores(model, data, modelName);
    }

    public static void findScores(Classifier model, Instances data, String modelName) throws Exception {
        int[] splits = new int[]{20, 40, 60, 80, 100};

        // deleted functionality for train and test scores to run faster
        ArrayList<Double> cvScores = new ArrayList<Double>();

        Instances[] dataSplits = getSplits(data, splits.length);

        StringBuilder result = new StringBuilder();
        for (int i = 0; i < splits.length; i++) {
//          System.out.println("==================================");
//          System.out.println("Split: " + splits[i]);
            runTests(model, splits[i], dataSplits[i], result, cvScores);
        }

        for (int i = 0; i < cvScores.size(); i++) {
            double score = cvScores.get(i);
            System.out.println(score);
        }
        System.out.println(result.toString()); // To print out runtime

//        result.append("trainScores = " + getScoresString(trainScores) + "\n");
//        result.append("testScores = " + getScoresString(testScores) + "\n");
//        result.append("cvScores = " + getScoresString(cvScores) + "\n");
//        String resultStr = result.toString();

//        System.out.print(resultStr);

//        BufferedWriter writer = new BufferedWriter(new FileWriter(dir + "result.txt"));
//        writer.write(resultStr);
//        writer.close();
    }
}
