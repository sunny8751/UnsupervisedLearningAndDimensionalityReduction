import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.IndependentComponents;
import weka.filters.unsupervised.attribute.PrincipalComponents;

public class PCA {

    public Filter getFilter() {
        return new PrincipalComponents();
    }


    public void run(String dataset) throws Exception {
        Instances data = Utils.readDataFile("datasets/" + dataset + "/data.arff");

        PrincipalComponents filter = (PrincipalComponents) getFilter();
//        IndependentComponents ica = new IndependentComponents();

        Instances newData = Filter.useFilter(data, filter);
//        Instances newData = Filter.useFilter(data, ica);
    }
}
