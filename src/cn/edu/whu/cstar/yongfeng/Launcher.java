package cn.edu.whu.cstar.yongfeng;

import java.util.List;

import weka.core.Instance;

public class Launcher {

	public static void main(String[] args) {
		
		SampleReader srTrain = new SampleReader("files/x264.csv");
		SampleReader srTest = new SampleReader("files/x264-testing.csv");
//		sr.showSamples();
		
		List<Instance> training = srTrain.getSamples();
		List<Instance> testing = srTest.getSamples();
		
		CARTModel model = new CARTModel(training, testing);
		model.predictSamples();
	}

}
