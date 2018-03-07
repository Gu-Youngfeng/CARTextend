package cn.edu.whu.cstar.yongfeng;

import java.util.List;

import weka.core.Instance;

public class CARTModel {
	/** training set */
	private List<Instance> training;
	/** testing set */
	private List<Instance> testing;
	/** CART tree model */
	private TreeNode cartTree;
	
	CARTModel(List<Instance> train, List<Instance> test){
		training = train;
		testing = test;
		
		CARTree tree = new CARTree(training);
		cartTree = tree.buildModel();
	}
	
	/***
	 * To predict one sample based on CARTModel
	 * @param sample
	 * @return performance
	 */
	private double predictOneSample(Instance sample){
		double performance = -1.0d;
		TreeNode pointer = cartTree;
		
		while(pointer != null){
			int node = pointer.getNodeName();
			if(pointer.getLeftChild() == null && pointer.getRightChild() == null){
				// this is a performance
				performance = node*(-1);
//				System.out.println("[predicted performance]: " + performance);
				break;
			}else{
				// this is a feature index
				double featureIndex = sample.value(node);
				if(featureIndex > 0){
					pointer = pointer.getRightChild();
				}else{
					pointer = pointer.getLeftChild();
				}
			}
		}
		
		return performance;		
	}
	
	public void predictSamples(){
		int correct = 0;
		int sum = testing.size();
		System.out.println("\n### Testing Results:");
		for(Instance ins: testing){
			double actualPerformance = ins.value(ins.numAttributes()-1);
			double predictedPerformance = predictOneSample(ins);
			System.out.println("[Actual]: " + actualPerformance + ", [Predicted]: " + predictedPerformance);
			if(actualPerformance == predictedPerformance){				
				correct++;
			}
		}
		System.out.println("[Fault Rate]: " + (1-(correct*1.0/sum*1.0)));
	}
	
}
