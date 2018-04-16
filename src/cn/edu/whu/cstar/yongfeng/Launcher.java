package cn.edu.whu.cstar.yongfeng;

import java.util.List;
import weka.core.Instance;

public class Launcher {
	/** single data file path */
	public static final String DATA_PATH = "files/Apache.csv";
	/** size of training set */
	public static final int N = 29;
	/** main entry of experiments */
	public static void main(String[] args) {
		
		/** STEP 1: Read from the csv file **/
		SampleReader sr = new SampleReader(DATA_PATH);
//		sr.showSamples();
		
//		CARTree tr = new CARTree(sr.getSamples()); // Construct CART tree
//		tr.buildModel();
		
		/** STEP 2: Select N samples as training set, the other as testing set **/
		List<List<Instance>> ls2T = sr.generate2T(N);
		List<Instance> training = ls2T.get(0);
		List<Instance> testing = ls2T.get(1);
		
		/** STEP 3: Predict the result **/
		CARTModel model = new CARTModel(training, testing);
		model.predictSamples();
		
	}

}
