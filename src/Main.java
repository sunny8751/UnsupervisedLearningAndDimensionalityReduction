import weka.core.Instances;

public class Main {

    /**
     * For each learning algorithm, run grid search first to get the optimal parameters.
     * Then run findScores() with a new instance of the model, passing in the optimal parameters into the constructor,
     * to get the learning curve scores. After getting a results.txt for all the algorithms,
     * run plotAllResults.py to get a graph of the linear curve.
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        String wineDataset = "winequalitywhite";
        String germanDataset = "germancredit";

        String dataFile = "data";
        String icaUntrimmedFile = "ica_untrimmed_data";
        String icaTrimmedFile = "ica_trimmed_data";
        String pcaUntrimmedFile = "pca_untrimmed_data";
        String pcaTrimmedFile = "pca_trimmed_data";

        String rp1File = "rp1_trimmed_data";
        String rp2File = "rp2_trimmed_data";
        String rp3File = "rp3_trimmed_data";

        String igFile = "ig_trimmed_data";



        /**
         * K Means
         */
//        new KMeans().gridSearch(wineDataset, igFile, 1, 25);
//        System.out.println(new KMeans().run(wineDataset, pcaTrimmedFile, 3));

//        new KMeans().gridSearch(germanDataset, igFile, 1, 25);
//        System.out.println(new KMeans().run(germanDataset, dataFile, 3));

        /**
         * EM
         */
//        new ExpectationMaximization().gridSearch(wineDataset, igFile, 1, 25);
//        System.out.println(new ExpectationMaximization().run(wineDataset, dataFile, 3));

//        new ExpectationMaximization().gridSearch(germanDataset, igFile, 1, 25);
//        System.out.println(new ExpectationMaximization().run(germanDataset, dataFile, 3));


        // Wine Dataset ICA: WEKA GUI, using StudentFilter pacakge -> IndependentComponents
        // Wine Dataset PCA: WEKA GUI, Principal Components

        /**
         * PCA, ICA, RP, IG are used in WEKA GUI
         */

        /**
         * Neural Network for DR-reduced Dataset
         */
        // print out NN scores for data after DR
        String[] files = new String[] {dataFile, pcaTrimmedFile, icaTrimmedFile, rp1File, igFile};
//        for (String file : files) {
//            System.out.println("===============================");
//            System.out.println(file);
//            new NeuralNetwork(.1f, .7f, 700).findScores(wineDataset, file);
//        }

        /**
         * Neural Network for Clusterss as Features
         */
        // print out NN scores for data with clusters as features
        for (String file : files) {
            System.out.println("===============================");
            System.out.println(file);
            Utils.runNNWithClustersAsOnlyFeatures(file, .1f, .7f, 700);
        }
    }

}
