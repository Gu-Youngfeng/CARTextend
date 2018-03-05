package cn.edu.whu.cstar.yongfeng;
import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SampleReader {

	private List<Instance> arrIns;
	
	private String[] arrFeatures;
	
	SampleReader(String path){
		
		Instances ins;
		try {
			//// read from the path
			ins = DataSource.read(path);
			
			//// in case of null path
			if(ins == null){
				System.out.println("[ERROR]: CAN NOT READ FROM: " + path);
				return;
			}
			
			//// index setting
			ins.setClassIndex(ins.numInstances()-1);
			
			//// samples and features initialization
			arrIns = new ArrayList<Instance>();
			for(int i=0; i<ins.numInstances(); i++){
				arrIns.add(ins.instance(i));
			}
			
			arrFeatures = new String[ins.numAttributes()];
			for(int j=0; j<ins.numAttributes(); j++){
				arrFeatures[j] = ins.attribute(j).name();
			}
			
		} catch (Exception e) {
			System.out.println("[ERROR]: CAN NOT INITIALIZE SAMPLES.");
			e.printStackTrace();
		}	
	}
	
	/** Return samples read from the path */
	public List<Instance> getSamples(){
		return arrIns;
	}
	
	/** Return features array */
	public String[] getFeatures(){
		return arrFeatures;
	}
	
	/** Print samples read from the path*/
	public void showSamples(){
		for(int i=0; i<arrIns.size(); i++){
			for(int j=0; j<arrIns.get(i).numAttributes(); j++){
				System.out.print(" | " + arrIns.get(i).value(j) + " | ");
			}
			System.out.println("");
		}
	}
	
	public static void showSample(Instance ins){
		for(int i=0; i<ins.numAttributes(); i++){
			System.out.print(ins.attribute(i) + ", ");
		}
		System.out.println("");
	}

}
