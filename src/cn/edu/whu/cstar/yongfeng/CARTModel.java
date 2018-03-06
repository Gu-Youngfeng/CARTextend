package cn.edu.whu.cstar.yongfeng;

import java.util.List;

import weka.core.Instance;

public class CARTModel {

	private List<Instance> training;
	
	private List<Instance> testing;
	
	CARTModel(List<Instance> train, List<Instance> test, List<String> features){
		training = train;
		testing = test;
	}
	
	/***
	 * To predict one sample based on CARTModel
	 * @param sample
	 * @return performance
	 */
	@SuppressWarnings("unused")
	private double predict(Instance sample){
		double performance = -1.0d;
		
		return performance;		
	}
	
	/***
	 * To judge whether the sample is matched with the given pattern
	 * @param sample
	 * @param pattern
	 * @return
	 */
	private int isMatch(Instance sample, Instance pattern){
		int flag = 0;
		
		return flag;
	}
	
	private void buildTree(){
		CARTree tree = new CARTree(training);
		TreeNode rootNode = tree.buildModel();
		
	}
}
