package src;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import src.NuestroModelo.DistanceType;
import src.NuestroModelo.DistanceWight;
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
		 //Modelo.main(paramsM);

		 NuestroModelo nm;
		 double fm = 0;
		 for (int k=1;k<100; k+=1){
			 for(DistanceWight dw: DistanceWight.values()){
				 for (DistanceType dt: DistanceType.values()) {
					 System.out.format("k: %d | dType: %s | dwight: %s%n", k, dw.toString(), dt.toString());
					 Instances instancias = Lector.getLector().leerInstancias("ficheros/preTrain.arff");
					 Instances instanciasparaclasificar = Lector.getLector().leerInstancias("ficheros/preDev.arff");
					 nm= new NuestroModelo(k,dw,dt, 1);
					 for (int i=0;i<instanciasparaclasificar.numInstances();i++) {
						 nm.prepararInstancias(instancias, instanciasparaclasificar.get(i));
						 instanciasparaclasificar.get(i).setClassValue(nm.clasificarInstancia(instanciasparaclasificar.get(i),instancias));
					 }
					 nm.crearMatrixConfusion(instanciasparaclasificar, instancias);
					 fm = nm.calcularMediciones(fm);
				}
			 }
		 }
		
	}

}
