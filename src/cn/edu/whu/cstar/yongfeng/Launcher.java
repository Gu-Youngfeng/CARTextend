package cn.edu.whu.cstar.yongfeng;

import java.util.List;

import weka.core.Instance;

public class Launcher {

	public static void main(String[] args) {
		SampleReader sr = new SampleReader("files/x264.csv");
//		sr.showSamples();
		
		List<Instance> lsSamples = sr.getSamples();
		String[] arrFeatures = sr.getFeatures();
		
		CARTree tree = new CARTree(lsSamples);
//		TreeNode crtTree = tree.buildModel();
		tree.printModel();
		
		

	}

}
