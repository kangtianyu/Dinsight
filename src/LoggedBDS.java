import java.io.File;
import java.io.FileNotFoundException;

import org.neuroph.core.data.BufferedDataSet;
import org.neuroph.core.data.DataSetRow;


public class LoggedBDS extends BufferedDataSet {

	private int n;

	public LoggedBDS(File file, int inputSize, int outputSize, String delimiter)
			throws FileNotFoundException {
		super(file, inputSize, outputSize, delimiter);
		n = 0;
	}
	
    @Override
    public DataSetRow next() {
    	if(++n%1000==0) System.out.println(n);
    	return super.next();
    }

}
