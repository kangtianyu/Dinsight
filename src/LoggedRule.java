import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.learning.LMS;

public class LoggedRule extends LMS {
	@Override
    public void doLearningEpoch(DataSet trainingSet) {
		super.doLearningEpoch(trainingSet);
		System.out.println(this.currentIteration+" "+this.getTotalNetworkError());
	}
}
