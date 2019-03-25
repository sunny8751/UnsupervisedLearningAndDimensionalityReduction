import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class KMeans extends ClustererAlgorithm {

    @Override
    protected Clusterer getClusterer(int k) throws Exception {
        SimpleKMeans clusterer = new SimpleKMeans();
        clusterer.setPreserveInstancesOrder(true);
//        clusterer.setMaxIterations(10000);
        clusterer.setNumClusters(k);
        clusterer.setSeed(100);
        return clusterer;
    }


    @Override
    public ClusterResult run(Instances data, Instances dataWithoutClass, int k) throws Exception {
        long startTime = System.currentTimeMillis();
        SimpleKMeans clusterer = (SimpleKMeans) getClusterer(k);
        clusterer.buildClusterer(dataWithoutClass);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(data);
        String result = eval.clusterResultsToString();
//        System.out.println(result);
        int startIndex = result.indexOf("\t", result.indexOf("Incorrectly clustered instances :\t") + "Incorrectly clustered instances :\t".length());
        double percentIncorrect = Double.parseDouble(result.substring(startIndex, result.indexOf("%", startIndex)).trim());
        return new ClusterResult("Sum of Squared Error", clusterer.getSquaredError(), percentIncorrect, (System.currentTimeMillis() - startTime) / 1000f);
    }

}
