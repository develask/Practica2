package src;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.core.Instance;
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
		 NuestroModelo nm= new NuestroModelo();
		 
		 Instances instancias = Lector.getLector().leerInstancias("ficheros/diabetes.arff");
		 Instances instanciasnoclasificadas = Lector.getLector().leerInstancias("ficheros/falta.arff");
		 for (Instance instancia : instanciasnoclasificadas) {
			 nm.prepararInstancias(instancias, instancia);
			 nm.clasificarInstancia(instancia);
		 }
		 // aqui faltaria coger las verdaderas y falsas y sacar TPR FPR manualmente y luego comparar.
		 
	}

}
