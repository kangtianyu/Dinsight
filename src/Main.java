import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.nnet.ElmanNetwork;
import org.neuroph.nnet.Instar;
import org.neuroph.nnet.JordanNetwork;

public class Main {

	private static ArrayList<String> features;
	private static int trainSampleNum;
	private static ArrayList<String[]> sample_data;
	private static HashMap<String,Double> sid;
	private static HashMap<String,Double> stepName;
	private static HashMap<String, Double> probName;

	public static void main(String[] args) {
		String workingDirectory = System.getProperty("user.dir");
		int maxIter = 10;
		try {
			// Read train data
			File trainData = new File(workingDirectory + "/data/algebra_2005_2006_train.txt");
			BufferedReader br = new BufferedReader(new FileReader(trainData));
			String str = br.readLine();
			features = new ArrayList<String>(Arrays.asList(str.split("\t")));
//			for(String ts:features)	System.out.println(ts);
			trainSampleNum = 0;
			sample_data = new ArrayList<String[]>();
			sid = new HashMap<String,Double>();
			stepName = new HashMap<String,Double>();
			probName = new HashMap<String,Double>();
			int out = features.indexOf("Correct First Attempt");
			int f1 = features.indexOf("Anon Student Id");
			int f2 = features.indexOf("Step Name");
			int f3 = features.indexOf("KC(Default)");
			System.out.println("Reading data from file");
			while((str = br.readLine())!=null){
				String[] tmps = str.split("\t",features.size());
				sample_data.add(tmps);
				if(!sid.containsKey(tmps[f1])) sid.put(tmps[f1], (double) sid.size());
				if(!stepName.containsKey(tmps[f2])) stepName.put(tmps[f2], (double) stepName.size());
				if(!probName.containsKey(tmps[f3])) probName.put(tmps[f3], (double) probName.size());
//				System.out.println(sample_data.get(trainSampleNum)[features.indexOf("Correct First Attempt")]);
//				System.out.println(sample_data.get(trainSampleNum).length+"");
//				System.out.println(sample_data.get(trainSampleNum)[0]+" "+sample_data.get(trainSampleNum)[features.indexOf("KC(Default)")]);
//				if(trainSampleNum%1000==0)System.out.println(trainSampleNum+"");
				trainSampleNum++;
			}
//			System.out.println(trainSampleNum+"");
			System.out.println(probName.size()+"");
			br.close();
			System.out.println("Data closed");
			
			// prepare neural network
//			NeuralNetwork neuralNetwork = new Outstar(sid.size()+probName.size());
//			NeuralNetwork neuralNetwork = new SupervisedHebbianNetwork(sid.size()+probName.size(),1);
//			NeuralNetwork neuralNetwork = new ElmanNetwork(sid.size()+probName.size(),1,2,1);
			NeuralNetwork neuralNetwork = new JordanNetwork(sid.size()+probName.size(),1,1,1);
			System.out.println("neuralNetwork builded");
//			neuralNetwork.setLearningRule(new LoggedRule());
			((IterativeLearning)neuralNetwork.getLearningRule()).setMaxIterations(maxIter);
			System.out.println("maxIter setted");

			// create training set
//			DataSet trainingSet = new DataSet(sid.size()+probName.size(), 1);
			// add training data to training set
			System.out.println("Building training set");
			int count=0;
			trainData = new File(workingDirectory + "/data/train.txt");
			if(!trainData.exists()){
				FileWriter fw = new FileWriter(trainData);
				BufferedWriter bw = new BufferedWriter(fw);
				for(String[] sample:sample_data){
	//				double[] input = new double[sid.size()+probName.size()];
	//				Arrays.fill(input,0.0);
	//				input[(int)((double)sid.get(sample[f1]))] = 1.0;
	//				input[(int)((double)sid.size()+probName.get(sample[f3]))] = 1.0;
					for(int i=0;i<sid.size()+probName.size();i++){
						if(i>0) bw.write(",");
						if(i==(int)((double)sid.get(sample[f1]))||
								(i==(int)((double)sid.size()+probName.get(sample[f3])))){
							bw.write("1");
						}else{
							bw.write("0");
						}
					}				
					bw.write(","+sample[out]);
					if(count<sample_data.size()-1) bw.write("\n");
	//				trainingSet.addRow(new DataSetRow (input,
	//						new double[]{Double.valueOf(sample[out])}));
					if(count%1000==0) System.out.println(count+"");
					count++;
				}
				bw.close();
			}

			DataSet trainingSet = new LoggedBDS(trainData,sid.size()+probName.size(),1,",");
			// Train neural network
			// learn the training set
			System.out.println("Training Start");
			System.out.println("Training Set size: " + trainingSet.size());
			neuralNetwork.learn(trainingSet);
			System.out.println("Testing result");
			// test result
			ArrayList<Double> result = new ArrayList<Double>();
			double output,gtoutput,error,rmse = 0;
			count = 0;
			int bigerr = 0;
			for(String[] sample:sample_data){
				double[] test = new double[sid.size()+probName.size()];
				Arrays.fill(test,0.0);
				test[(int)((double)sid.get(sample[f1]))] = 1.0;
				test[(int)((double)sid.size()+probName.get(sample[f3]))] = 1.0;
				neuralNetwork.setInput(test);
				neuralNetwork.calculate();
				output = neuralNetwork.getOutput()[0];
				gtoutput = Double.valueOf(sample[out]);
				error = (output-gtoutput)*(output-gtoutput);
				System.out.println(sid.get(sample[f1])+" "+probName.get(sample[f3])+" "+output+" "+gtoutput+" "+error);
				result.add(error);
				rmse += error;
				if(error>0.25) bigerr++;
				count++;
			}
			rmse = Math.sqrt(rmse/result.size());
			System.out.println("Testing End.(mean RMSE="+rmse+")" + bigerr + "/" + count);
			// save the trained network into file
//			neuralNetwork.save(workingDataset+"_"+gt+".nnet");
			
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
