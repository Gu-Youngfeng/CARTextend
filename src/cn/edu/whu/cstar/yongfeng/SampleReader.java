package cn.edu.whu.cstar.yongfeng;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SampleReader {
	
	private Instances allSamples;

	private List<Instance> arrIns;
	
	private String[] arrFeatures;
	
	SampleReader(String path){
		
		try {
			//// read from the path
			allSamples = DataSource.read(path);
			
			//// in case of null path
			if(allSamples == null){
				System.out.println("[ERROR]: CAN NOT READ FROM: " + path);
				return;
			}
			
			//// index setting
			allSamples.setClassIndex(allSamples.numInstances()-1);
			
			//// samples and features initialization
			arrIns = new ArrayList<Instance>();
			for(int i=0; i<allSamples.numInstances(); i++){
				arrIns.add(allSamples.instance(i));
			}
			
			arrFeatures = new String[allSamples.numAttributes()];
			for(int j=0; j<allSamples.numAttributes(); j++){
				arrFeatures[j] = allSamples.attribute(j).name();
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
	
	/***
	 * <p>To generate the <b>Training</b> and <b>Testing</b> sets in all samples.</p> 
	 * <p>Note that we first randomize the samples and then select the first <b>num</b> samples as training set, 
	 * while choose the remaining samples as testing set.</p>
	 * @param num size of training set
	 * @return <b>lsAll</b> lsAll.get(0) is the Training set, lsAll.get(1) is the Testing set.
	 */
	public List<List<Instance>> generate2T(int num){
		List<List<Instance>> lsAll = new ArrayList<List<Instance>>();
		// randomize the samples
		allSamples.randomize(new Random(System.currentTimeMillis()));
		
		List<Instance> lsTesting = new ArrayList<Instance>();
		List<Instance> lsTraining = new ArrayList<Instance>();
		
		if(num >= allSamples.size()){
			System.out.println("[ERROR]: num CANNOT bigger than sample size.");
			System.out.println("[num]:" + num + ", [sample size]: " + allSamples.size());
			return null;
		}
		
		for(int i=0; i<num; i++){ // take the first num samples as training set
			lsTraining.add(allSamples.instance(i));
		}
		for(int j=num; j<allSamples.size(); j++){ // take the remaining samples as testing set
			lsTesting.add(allSamples.instance(j));
		}
		
		lsAll.add(lsTraining);
		lsAll.add(lsTesting);
		
		return lsAll;
	}

}
