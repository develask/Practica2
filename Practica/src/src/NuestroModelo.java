package src;

import java.util.ArrayList;
import java.util.Collections;

import distance.*;
import weka.core.Instance;
import weka.core.Instances;

public class NuestroModelo { 
	
	private ArrayList<Double> lista = new ArrayList<Double>();
	//no he encontrado ninguna forma de enlazar una distancia y una instancia, hay que mirarlo.
    /**
     * 1- Numero de vecinos para analizar
     * */
	protected int kNN;
	/**
	 * 1- "No distance weighting"
	 * 2- "Weight by 1/distance"
	 * 3- "Weight by 1-distance"
	 */
	protected int distanceWeighting;
    /**
     * 1- KNN con rechazo
     * 2- Distancia media
     * 3- Distancia Minima
     * 4- Pesado de casos seleccionados
     * 5- Pesado de variable sadfasdfsaf
     * */
	protected int NNSearch;
	
	/**
	 * Constructor para crear el modelo KNN.
	 * @param KNN numero de vecionos a analizar. [1:]
	 * @param distance Tipo de distancia a analizar: [(1: Manhattan), (2: Euclídea), (3: Minkowski)]
	 * @param searchAlgoritm Algoritmo de busqueda: [1:5]
	 */
	public NuestroModelo(int KNN, int distance, int searchAlgoritm){
	    this.setKNN(KNN);
	    this.setDistanceWeighting(distance);
	    this.setNearestNeighbourSearchAlgorithm(searchAlgoritm);
	}

	public void setDistanceWeighting(int i){
		if (i<1 || i>3) i = 1;
		this.distanceWeighting=i;
		
	}
	public int getDistanceWeighting(){
		return this.distanceWeighting;
	}

	public int getKNN(){
		return kNN;
	}

	public void setKNN(int k) {
		if (k<1) k = 1;
		this.kNN=k;
	}
	
	public int getNearestNeighbourSearchAlgorithm() {
		return NNSearch;
	}
	
	public void setNearestNeighbourSearchAlgorithm(int i) {
		if (i<1 || i>5) i = 1;
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
	
	private double calcularDistancia(Instance instancia,Instance instanciaaclasificar) {
		int metodo = this.getNearestNeighbourSearchAlgorithm();
		int numAtr = instanciaaclasificar.numAttributes();
		Distance dis;
		switch (metodo) {
		case 1:
			dis = new Manhattan(numAtr);
			break;
		case 2:
			dis = new Euclidea(numAtr);
			break;
		case 3:
			dis = new Minkowski(numAtr);
			break;
		default:
			return 0.00;
		}
		for(int i=1;i<=numAtr;i++){
			dis.setAtributeDist(instancia.value(i), instanciaaclasificar.value(i));
		}
		return dis.getDistance();
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
		}else if(metodo == 4){
			for(int i=1;i<=numerovecinos;i++){
				
			}
		}else if(metodo == 5){
			for(int i=1;i<=numerovecinos;i++){
				
			}
		}
	}
}	
