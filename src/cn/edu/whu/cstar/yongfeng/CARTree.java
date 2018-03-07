package cn.edu.whu.cstar.yongfeng;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import weka.core.Instance;

public class CARTree {
	
	/** all valid configurations */
	private List<Instance> samples;
//	/** features in configuration */
	private int[] features;
	/** 1st parameter in CART algorithm */
	private double minBucket;
	/** 2nd parameter in CART algorithm */
	private double minSplit;
	
	private List<Integer> featureList;
	
	/***
	 * <p>Initializing the CART with <b>{@link#samples}</b> and <b>{@link#features}</b>. 
	 * In addition, the parameter of <b>minbucket</b> and <b>minsplit</b> will also be defined.</p>
	 * @param s samples
	 * @param f features
	 */
	CARTree(List<Instance> s){
		
		// Input of CART
		samples = s;
		int numAttr = s.get(0).numAttributes();
		features = new int[numAttr];
		for(int i=0; i<numAttr; i++){
			features[i] = i;
		}
//		features = f;
		
		// Parameters setting of CART
		int lenSample = s.size();
		if(lenSample > 100){
			minSplit = Math.floor(Math.floor(lenSample*1.0/10*1.0)+0.5);
			minBucket = Math.floor(minSplit*1.0/2*1.0);
		}else{
			minBucket = Math.floor(Math.floor(lenSample*1.0/10*1.0)+0.5);
			minSplit = 2*minBucket;
		}
		minBucket = minBucket>=2?minBucket:2;
		minSplit = minSplit>=4?minSplit:4;
		
		System.out.println("### Experimental Parameters:");
		System.out.println("[sample size]: " + samples.size());
		System.out.println("[minSplit   ]: " + minSplit);
		System.out.println("[minBucket  ]: " + minBucket);
		
		featureList = new ArrayList<Integer>();
	}
	
	/***
	 * <p>return the CARTree model built by the training samples.</p>
	 * @return TreeNode
	 */
	public TreeNode buildModel(){
		
		System.out.println("\n### Building CARTree Model:");
		goIteration(samples, features);
//		for(Integer node: featureList) System.out.print(node + " - ");
		TreeNode rootNode = constructTree(featureList);
		
		System.out.println("\n### Construct of CARTree:");
		rootNode.showTree(rootNode);
		System.out.println("");
		
		return rootNode;
	}
	
	private void goIteration(List<Instance> S, int[] F){
		int index = getBetterSplitIndex(S, F);
		
		if(index == -1){
			System.out.println("[Leaf Node]:" + getAVEPerformance(S));
			featureList.add(0-(int)Math.round(getAVEPerformance(S))); // use Math.round()
		}else{
			int currentFeature = F[index];
			featureList.add(currentFeature);
			List<List<Instance>> lsSLR = getSplitTwoSamples(S, index);
//			System.out.println("[Left  S]:" + currentFeature);
			goIteration(lsSLR.get(0), F); // visit left SL
//			System.out.println("[Right S]:" + currentFeature);
			goIteration(lsSLR.get(1), F); // visit right SR 
		}
	} 
	
	private TreeNode constructTree(List<Integer> features){
		Stack<TreeNode> stNode = new Stack<TreeNode>();
//		LinkedList<> nodeStruct =
		if(features == null || features.size() == 0){
			System.out.println("[ERROR]: featureList is null!");
			return null;
		}
				
		List<TreeNode> lsNode = new ArrayList<TreeNode>();
		for(Integer featureName: features){
			TreeNode node = new TreeNode(featureName);
			lsNode.add(node);
		}
		
		TreeNode rootNode = lsNode.get(0);
		stNode.push(rootNode);
		
		for(int i=1; i<lsNode.size(); i++){
			TreeNode preNode = lsNode.get(i-1);
			TreeNode curNode = lsNode.get(i);
			
			if(preNode.getNodeName()>=0 && curNode.getNodeName()<0){
				// feature node -> leaf node
				preNode.setLeftChild(curNode);
//				System.out.println(preNode.getNodeName() + " Leftchild is " + curNode.getNodeName());

			}else if (preNode.getNodeName()>=0 && curNode.getNodeName()>=0){
				// feature node -> feature node
				preNode.setLeftChild(curNode);
				stNode.push(curNode);
//				System.out.println(preNode.getNodeName() + " Leftchild is " + curNode.getNodeName());
//				System.out.println("[PUSH]:" + curNode.getNodeName());
				
			}else if(preNode.getNodeName()<0 && curNode.getNodeName()>=0){
				// leaf node -> feature node
				TreeNode tempNode = stNode.pop();
				tempNode.setRightChild(curNode);
				stNode.push(curNode);
//				System.out.println("[POP]:" + tempNode.getNodeName());
//				System.out.println(tempNode.getNodeName() + " Rightchild is " + curNode.getNodeName());				
				
			}else{
				// leaf node -> leaf node
//				System.out.println(tempNode.getNodeName() + " Rightchild is " + curNode.getNodeName());				
				TreeNode tempNode = stNode.pop();				
				tempNode.setRightChild(curNode);
//				System.out.println("[POP]:" + tempNode.getNodeName());
				
			}
		}
		
//		System.out.println("[STACK]: is " + stNode.isEmpty());
		
		return rootNode;
	}
	
	/***
	 * <p>To split sample set <b>S</b> with the given feature list <b>F</b> for one time.
	 * This function will return the best split feature index of <b>F</b>, 
	 * note that if the split meets stop criteria the return value will be -1.</p>
	 * @param S
	 * @param F
	 * @return betterSplitIndex the best split feature index.
	 */
	private int getBetterSplitIndex(List<Instance> S, int[] F){
		int betterSplitIndex = -1;
		
		if(S.size() > minSplit){ // stop further split
			
			double[] arrSel = new double[F.length-1];
			
			for(int i=0; i<F.length-1; i++){ // for each feature
				
				/** split S into SL and SR */
				List<Instance> SL = new ArrayList<Instance>(); // left node of S
				List<Instance> SR = new ArrayList<Instance>(); // right node of S
				for(int j=0; j<S.size(); j++){
					Instance temp = S.get(j);					
					if(temp.value(i) < 0.5){ // attribute value == 0
						SL.add(temp);
					}else{ // attribute value == 1
						SR.add(temp);
//						System.out.println("[>0.5]: " + temp.value(j));
					}
				}
//				System.out.println("[FN]: " + F[i] + ", [SL]: " + SL.size() + ", [SR]: " + SR.size());
				if(SL.size() < minBucket || SR.size() < minBucket){ // stop further split		
					arrSel[i] = Double.MAX_VALUE;
//					System.out.println("[STOP]: minBucket is too small. " + F[i]);
				}else{
					/** calculate the squared error loss of both*/
					arrSel[i] = getSquaredErrorLoss(SL, SR);
//					System.out.println("[arrSel]: " + arrSel[i] + "\n");
				}
			}
			
//			for(int k=0; k<arrSel.length; k++){
//				System.out.print(arrSel[k] + ", ");
//			}
//			System.out.println("");
			betterSplitIndex = getMinValue(arrSel); // which feature can get the minimal squared error loss  
			System.out.println("[feature]: " + F[betterSplitIndex] + ", [index]: " + betterSplitIndex + ", [loss]: " + arrSel[betterSplitIndex]);
			
		}else{
//			System.out.println("[STOP]: minSplit is too small.");
//			System.out.println("[STOP AT AVE]:" + getAVEPerformance(S));			
			return -1;
		}
		
		return betterSplitIndex;
	}
	
	private double getSquaredErrorLoss(List<Instance> SL, List<Instance> SR){	
		
		double sumSL = 0.0d;
		double sumSR = 0.0d;
		
		for(int i=0; i<SL.size(); i++){
			double tempPerformance = SL.get(i).value(SL.get(i).numAttributes()-1);
			sumSL += tempPerformance;
//			System.out.print(SL.get(i).value(SL.get(i).numAttributes()-1) + ", ");
		}
//		System.out.println("");
		
		for(int i=0; i<SR.size(); i++){
			double tempPerformance = SR.get(i).value(SR.get(i).numAttributes()-1);
			sumSR += tempPerformance;
//			System.out.print(SR.get(i).value(SR.get(i).numAttributes()-1) + ", ");
		}
//		System.out.println("");
		
		double aveSL = sumSL/SL.size();
		double aveSR = sumSR/SR.size();
		
		double deltSL = 0.0d;
		double deltSR = 0.0d;
		
		for(int i=0; i<SL.size(); i++){
			double tempPerformance = SL.get(i).value(SL.get(i).numAttributes()-1);
			deltSL += Math.pow((tempPerformance - aveSL), 2.0d);
		}
		
		for(int i=0; i<SR.size(); i++){
			double tempPerformance = SR.get(i).value(SR.get(i).numAttributes()-1);
			deltSR += Math.pow((tempPerformance - aveSR), 2.0d);
		}
		
		double sum = deltSL + deltSR;
		
		return sum;
	}

	private int getMinValue(double[] arrSel){
		double min = arrSel[0];
		int minIndex = 0;
		for(int i=0; i<arrSel.length; i++){
			if(min > arrSel[i]){
				min = arrSel[i];
				minIndex = i;
			}
		}
		
		return minIndex;
	}
	
	private List<List<Instance>> getSplitTwoSamples(List<Instance> S, int index){
				
		/** split S into SL and SR */
		List<Instance> SL = new ArrayList<Instance>(); // left node of S
		List<Instance> SR = new ArrayList<Instance>(); // right node of S
		for(int j=0; j<S.size(); j++){
			Instance temp = S.get(j);					
			if(temp.value(index) < 0.5){ // attribute value == 0
				SL.add(temp);
			}else{ // attribute value == 1
				SR.add(temp);
			}
		}
				
		List<List<Instance>> lsSLR = new ArrayList<List<Instance>>();
		lsSLR.add(SL);
		lsSLR.add(SR);
		System.out.println("[SL]" + SL.size() + "[SR]" + SR.size());
		
		return lsSLR;
	}
	
	private double getAVEPerformance(List<Instance> S){

		double sum = 0.0d;
		if(S == null || S.size() == 0){
			return -1;
		}
		for(int i=0; i<S.size(); i++){
			double tempPer = S.get(i).value(S.get(i).numAttributes()-1);
//			System.out.print(tempPer + ", ");
			sum += tempPer;
		}
//		System.out.println("");
		double ave = (sum*1.0)/(S.size()*1.0);
		
		return ave;
	}
	
}
