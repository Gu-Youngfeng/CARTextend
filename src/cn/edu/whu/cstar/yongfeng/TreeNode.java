package cn.edu.whu.cstar.yongfeng;

import weka.core.Instance;

public class TreeNode {

	private TreeNode leftChild;
	private TreeNode rightChild;
	
	private Integer nodeName;
	
	TreeNode(Integer name){
		nodeName = name;
		leftChild = null;
		rightChild = null;
	}
	
	public void setLeftChild(TreeNode left){	
		leftChild = left;
	}
	
	public void setRightChild(TreeNode right){
		rightChild = right;
	}
	
	public TreeNode getLeftChild(){
		return leftChild;
	}
	
	public TreeNode getRightChild(){
		return rightChild;
	}
	
	public Integer getNodeName(){
		return nodeName;
	}
	
	public void showTree(TreeNode node){
		
//		System.out.println("");
		
		if(node != null){
			System.out.print(node.getNodeName());
			if(node.leftChild != null){
				System.out.print(" {");
				showTree(node.leftChild);
			}
				
			if(node.rightChild != null){
				System.out.print(", ");
				showTree(node.rightChild);
				System.out.print("} ");
			}
				
		}		
	}
	
}
