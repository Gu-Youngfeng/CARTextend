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
	
	/***
	 * <p>To build the CART Evaluation model by specify the <b>train</b> and <b>test</b> set.</p>
	 * @param train training set
	 * @param test testing set
	 */
	CARTModel(List<Instance> train, List<Instance> test){
		/** STEP 1: specify the train and test set*/
		training = train;
		testing = test;
		/** STEP 2: build the CART model by using train set*/
		CARTree tree = new CARTree(training);
		cartTree = tree.buildModel();
	}
	
	/***
	 * <p>To predict one sample based on CART model</p>
	 * <p>Technically, the given sample will go through the CART tree 
	 * and get the predict value (leaf value)</p>
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
	
	/***
	 * <p>Based on the CART model built on {@link#training}, and make prediction on {@link#testing}.</p>
	 * <p>The output measurements includes <b>Actual Value</b> and <b>Predicted Value</b>, and also <b>MMRE</b>.</p>
	 */
	public void predictSamples(){
		double correct = 0;
		int sum = testing.size();
		System.out.println("\n### Testing Results:");
		for(Instance ins: testing){
			double actualPerformance = ins.value(ins.numAttributes()-1);
			double predictedPerformance = predictOneSample(ins);
			double delt = Math.abs(actualPerformance - predictedPerformance)*1.0/(actualPerformance*1.0);
			correct += delt;
			System.out.println("[Actual]: " + actualPerformance + ", [Predicted]: " + predictedPerformance + ", [delt]: " + delt);
			
		}
		correct = correct*1.0/sum*1.0;
		System.out.println("[Fault Rate]: " + (1-correct));
	}
	
}
