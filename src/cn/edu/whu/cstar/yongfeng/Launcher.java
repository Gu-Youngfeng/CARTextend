package cn.edu.whu.cstar.yongfeng;

import java.util.List;
import weka.core.Instance;

public class Launcher {

	public static void main(String[] args) {
		
		SampleReader sr = new SampleReader("files/x264.csv");
//		sr.showSamples();
		
		List<List<Instance>> ls2T = sr.generate2T(15);
		List<Instance> training = ls2T.get(0);
		List<Instance> testing = ls2T.get(1);
		
		CARTModel model = new CARTModel(training, testing);
		model.predictSamples();
		
	}

}
