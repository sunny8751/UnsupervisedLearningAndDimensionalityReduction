import weka.clusterers.Clusterer;
import weka.core.Instances;

import java.util.Random;

public abstract class ClustererAlgorithm {

    public static class ClusterResult {
        public double runtime;
        public double error;
        public double incorrectlyClusteredPercent;
        private String errorName;

        public ClusterResult(String errorName, double error, double incorrectlyClusteredPercent, double runtime) {
            this.errorName = errorName;
            this.error = error;
            this.incorrectlyClusteredPercent = incorrectlyClusteredPercent;
            this.runtime = runtime;
        }
        @Override
        public String toString() {
            return errorName + ": " + error + "\tIncorrectly Clustered: " + incorrectlyClusteredPercent + "%\truntime: " + runtime;
        }
    }

    public void gridSearch(String dataset, String dataFile, int low, int high) throws Exception {
        Instances data = Utils.readDataFile("datasets/" + dataset + "/" + dataFile + ".arff");
        Instances dataWithoutClass = Utils.getWithoutClass(data);

        System.out.println("k\terror\t% Incorrectly clustered instances");
        for (int k = low; k <= high; k++) {
            ClusterResult result = run(data, dataWithoutClass, k);
            System.out.println(k + "\t" + result.error + "\t" + result.incorrectlyClusteredPercent);
//            System.out.println(result.error + "\t" + result.incorrectlyClusteredPercent);
        }
    }

    public ClusterResult run(String dataset, String dataFile, int k) throws Exception {
        Instances data = Utils.readDataFile("datasets/" + dataset + "/" + dataFile + ".arff");
        Instances dataWithoutClass = Utils.getWithoutClass(data);
        return run(data, dataWithoutClass, k);
    }

    protected abstract Clusterer getClusterer(int k) throws Exception;

    protected abstract ClusterResult run(Instances data, Instances dataWithoutClass, int k) throws Exception;
}
