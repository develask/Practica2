package src;

import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.AdditionalMeasureProducer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

public class NuestroModelo { 
	
	private ArrayList<Double> lista = new ArrayList<Double>();
	//no he encontrado ninguna forma de enlazar una distancia y una instancia, hay que mirarlo.
    protected int kNN;
    /**
     * 1- Numero de vecinos para analizar
     * */
    protected int distanceWeighting;
	/**
	 * 1- "No distance weighting"
	 * 2- "Weight by 1/distance"
	 * 3- "Weight by 1-distance"
	 */
    protected int NNSearch;
    /**
     * 1- KNN con rechazo
     * 2- Distancia media
     * 3- Distancia Minima
     * 4- Pesado de casos seleccionados
     * 5- Pesado de variables
     * */
	public NuestroModelo(){
	    this.setKNN(1);
	    this.setDistanceWeighting(1);
	    this.setNearestNeighbourSearchAlgorithm(1);
	}

	public void setDistanceWeighting(int i) {
		this.distanceWeighting=i;
		
	}
	public int getDistanceWeighting(){
		return this.distanceWeighting;
	}

	public int getKNN(){
		return kNN;
	}

	public void setKNN(int k) {
		this.kNN=k;
	}
	
	public int getNearestNeighbourSearchAlgorithm() {
		return NNSearch;
	}
	
	public void setNearestNeighbourSearchAlgorithm(int i) {
		NNSearch = i;
	}	 
   	
  
	public void prepararInstancias(Instances instancias, Instance instancia){
		double distancia=0.00;
		lista = new ArrayList<Double>();
		for(int j=1;j<=instancias.numInstances();j++){
			distancia=this.calcularDistancia(instancias.get(j),instancia);
			lista.add(distancia);
		}
		Collections.sort(lista);
	}
	private int calcularDistancia(Instance instancia,Instance instanciaaclasificar) {
		int metodo = this.getNearestNeighbourSearchAlgorithm();
		if (metodo == 1){
			for(int i=1;i<=instanciaaclasificar.numAttributes();i++){
				//sacar normal (x1-x2)^2 + ...
				instancia.attribute(i).
			}
		}else if(metodo == 2){
			//1/normal
			
		}else if(metodo == 3){
			// 1- normal
		}
		return 0.00;
	}

	public void clasificarInstancia(Instance NoClasificada){
		int numerovecinos = this.getKNN();
		int metodo = this.getNearestNeighbourSearchAlgorithm();
		//recorreremos el array hasta el numero de k en instancias
		if (metodo == 1){
			for(int i=1;i<=numerovecinos;i++){
				
			}
		}else if(metodo == 2){
			for(int i=1;i<=numerovecinos;i++){
				
			}
		}else if(metodo == 3){
			for(int i=1;i<=numerovecinos;i++){
				
			}
		}else if(metodo == 2){
			for(int i=1;i<=numerovecinos;i++){
				
			}
		}else if(metodo == 3){
			for(int i=1;i<=numerovecinos;i++){
				
			}
		}
	}
}	
