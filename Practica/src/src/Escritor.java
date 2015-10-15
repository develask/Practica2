package src;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;

import src.NuestroModelo.DistanceWight;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;


public class Escritor {

	private static Escritor miEscritor=null;
	
	private Escritor() {
		
	}		
	
	public static Escritor getEscritor(){
		if(Escritor.miEscritor==null){
			Escritor.miEscritor = new Escritor();
		}
		return Escritor.miEscritor;
	}
	
	public void escribirInstancias(Instances instancias, String path){
		File output= new File(path);
		try {
			output.createNewFile();
			BufferedWriter fw= new BufferedWriter(new FileWriter(output));
			fw.write(instancias.toString());
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void hacerFicheroIBk(String path, Evaluation evaluator, IBk estimador, int kMaxExhaustiba, String tipoEstimacion,Boolean nuevo){
		 try {
			
			String matriz = evaluator.toMatrixString();
			double fmeasureMedia = evaluator.weightedFMeasure();
			double precision = evaluator.weightedPrecision();			
			double recall = evaluator.weightedRecall();
			double roc = evaluator.weightedAreaUnderROC();
			double accu =evaluator.pctCorrect();
			double fmeasureV1 = evaluator.fMeasure(0);
			double fmeasureV2 = evaluator.fMeasure(1);
			
			try {
				FileWriter fw = new FileWriter(path,nuevo);
				fw.write("\n******************************************************\n");
				fw.write("\n****"+tipoEstimacion+"****\n");
				fw.write("\n******************************************************\n");
				fw.write("F-Measure Batazbestekoa: " + fmeasureMedia+"\n");
				fw.write("K Maximoa: " + estimador.getKNN()+"\n");
				fw.write("Distance Weighting: " + estimador.getDistanceWeighting()+"\n");
				fw.write("Revision: " + estimador.getRevision()+"\n");
				fw.write("Training Times: " + estimador.getNumTraining()+"\n");
				fw.write("Nearest Neighbour Searh Algorithm: " + estimador.getNearestNeighbourSearchAlgorithm()+"\n");
				fw.write("K Maximoa metodo ez exhaustiboarekin: " + kMaxExhaustiba+"\n");
				fw.write("Precision Batazbestekoa: " + precision+"\n");
				fw.write("Recall Batazbestekoa: " + recall+"\n");
				fw.write("ROC Area Batazbestekoa: " + roc+"\n");
				fw.write("F-Measure V1: " + fmeasureV1+"\n");
				fw.write("F-Measure V2: " + fmeasureV2+"\n");
				fw.write("Recall V1: " + evaluator.recall(0)+"\n");
				fw.write("Recall V2: " + evaluator.recall(1)+"\n");
				fw.write("Precision V1: " + evaluator.precision(0)+"\n");
				fw.write("Precision V2: " + evaluator.precision(1)+"\n");
				fw.write("Correctly Classified Instances: " + accu);
				fw.write("\n" + matriz);
				fw.write("\n******************************************************\n");
				fw.close();
			} catch (IOException e) {
				File f = new File(path);
				f.mkdirs();
				hacerFicheroIBk(path, evaluator, estimador, kMaxExhaustiba, tipoEstimacion, nuevo);
			}
		} catch (Exception e) {
				e.printStackTrace();
		}
	}

	public void hacerFicheroNuestroModelo(String path,int kNN,DistanceWight distanceWeighting,int nearestNeighbourSearchAlgorithm,float precision,float recall,float accuracy,float fmeasure,int tP,int tN,int fP,int fN,boolean nuevo){
	
		try {
				
			try {
				FileWriter fw= new FileWriter(path, nuevo);
				fw.write("\n******************************************************\n");
				fw.write("\n****Estimacion Nuestro modelo****\n");
				fw.write("\n******************************************************\n");
				fw.write("K Maximoa: " + kNN+"\n");
				fw.write("Distance Weighting: " + distanceWeighting+"\n");
				fw.write("Nearest Neighbour Searh Algorithm: " + nearestNeighbourSearchAlgorithm+"\n");
				fw.write("---------------------------------");
				fw.write("    Precision = "+precision+"    ");
				fw.write("    Recall = "+recall+"          ");
				fw.write("    Accuracy = "+accuracy+"      ");
				fw.write("    F-Measure = "+fmeasure+"     ");
				fw.write("---------------------------------");
				fw.write("------------Matriz De Confusión---------------------");
				fw.write("----------------------------------");
				fw.write("---|---------|-------------------");
				fw.write("---|-TP="+tP+"---|-FP="+fP+"--------");
				fw.write("---|---------|-------------------");
				fw.write("---|-TN="+tN+"---|-FN="+fN+"--------");
				fw.write("---|---------|-------------------");
				fw.close();
				 
		} catch (IOException e) {
			File f = new File(path);
			f.mkdirs();
			hacerFicheroNuestroModelo(path, kNN, distanceWeighting, nearestNeighbourSearchAlgorithm, precision, recall, accuracy, fmeasure, tP, tN, fP, fN , nuevo);
		}
		} catch (Exception e1) {
			e1.printStackTrace();
		}	
	}
	
	public void escribirModelo(String  path, Classifier cls){
	
		ObjectOutputStream oos;
		try {
			oos = new ObjectOutputStream(new FileOutputStream(path));
			oos.writeObject(cls);
			oos.flush();
			oos.close();
		} catch (FileNotFoundException e) {
			File f = new File(path);
			f.mkdirs();
			escribirModelo(path, cls);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
	}
}
