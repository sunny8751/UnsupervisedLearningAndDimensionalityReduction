import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class ExpectationMaximization extends ClustererAlgorithm {

    @Override
    protected Clusterer getClusterer(int k) throws Exception {
        EM clusterer = new EM();
//        clusterer.setMaxIterations(10000);
        clusterer.setNumClusters(k);
        clusterer.setSeed(100);
        return clusterer;
    }


    @Override
    public ClusterResult run(Instances data, Instances dataWithoutClass, int k) throws Exception {
        long startTime = System.currentTimeMillis();
        EM clusterer = (EM) getClusterer(k);
        clusterer.buildClusterer(dataWithoutClass);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(data);
        String result = eval.clusterResultsToString();
//        System.out.println(result);
        int startIndex = result.indexOf("\t", result.indexOf("Incorrectly clustered instances :\t") + "Incorrectly clustered instances :\t".length());
        double percentIncorrect = Double.parseDouble(result.substring(startIndex, result.indexOf("%", startIndex)).trim());
        return new ClusterResult("Log Likelihood", eval.getLogLikelihood(), percentIncorrect, (System.currentTimeMillis() - startTime) / 1000f);
    }

}
