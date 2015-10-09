package src;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.core.Instances;

public class Probador {
	
	public static void main(String[] args) {
		//Lehenengo datuak prosezatuko ditugu.
		String[] params = new String [5];
		 params[0] = "ficheros/diabetes.arff"; 
		 params[1] = "ficheros/preTrain.arff";
		 params[2] = "ficheros/preDev.arff";
		 params[3] = "30";
		 PreProcesador.main(params);
		 
		 //Creamos el modelo
		 String[] paramsM = new String [2];
		 paramsM[0] = "ficheros/preTrain.arff";
		 paramsM[1] = "ficheros/preDev.arff";
		 Modelo.main(paramsM);
		 //Primero el IBk
		 String[] paramsS = new String [3];
		 paramsS[0] = "modelos/IBkModel.model";
		 paramsS[1] = "ficheros/falta.arff";
		 paramsS[2] = "ficheros/TestPredictionsIBk.arff";
		 Clasificador.main(paramsS);
		//Ahora NuestroModelo
		 paramsS[0] = "modelos/NuestroModeloModel.model";
		 paramsS[1] = "ficheros/falta.arff";
		 paramsS[2] = "ficheros/TestPredictionsNuestroModelo.arff";
		 Clasificador.main(paramsS);
		 /*File output= new File("test.eval");
		 FileWriter fw = null;
		try {
			fw = new FileWriter(output);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		 Instances instancias = Lector.getLector().leerInstancias("ficheros/segment-test.arff");
		 
		 try {
			Evaluation evaluation = new Evaluation(instancias);
			
			IBk ib = (IBk) Lector.getLector().cargarModelo("modelos/IBkModel.model");
			NuestroModelo nuestro = (NuestroModelo) Lector.getLector().cargarModelo("modelos/NuestroModeloModel.model");
			
			evaluation.evaluateModel(ib, instancias);
			
			fw.append("Confusion Matrix :"+evaluation.toMatrixString()+"");
			fw.append(evaluation.toClassDetailsString());
			
			evaluation.evaluateModel(nuestro, instancias);
			
			fw.append("Confusion Matrix :"+evaluation.toMatrixString()+"");
			fw.append(evaluation.toClassDetailsString());
			
			
			fw.close();
		 } catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 
		 */
	}

}
