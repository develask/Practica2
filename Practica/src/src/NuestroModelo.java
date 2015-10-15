package src;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import distance.*;
import weka.core.Instance;
import weka.core.Instances;

public class NuestroModelo { 
	
	private double[][] lista;
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
	public enum DistanceWight {
		NoDistance, OneDivDistance, OneMinusDistance
	};
	
	protected DistanceWight distanceWeighting;
	
	public enum DistanceType {
		Manhattan, Euclidea, Minkowski
	};
	private DistanceType distance;
	private Minkowski distanceMethod;
    /**
     * 1- KNN con rechazo
     * 2- Distancia media
     * 3- Distancia Minima
     * 4- Pesado de casos seleccionados
     * 5- Pesado de variable sadfasdfsaf
     * */
	protected int NNSearch;
	
	
	private int tP;
	private int tN;
	private int fP;
	private int fN;
	
	/**
	 * Constructor para crear el modelo KNN.
	 * @param KNN numero de vecionos a analizar. [1:]
	 * @param distance Tipo de distancia a analizar: [(1: Manhattan), (2: Euclídea), (3: Minkowski)]
	 * @param searchAlgoritm Algoritmo de busqueda: [1:5]
	 */
	public NuestroModelo(int KNN, DistanceWight distanceW, DistanceType distanceT , int searchAlgoritm){
	    this.setKNN(KNN);
	    this.setDistanceWeighting(distanceW);
	    this.setDistance(distanceT);
	    this.setNearestNeighbourSearchAlgorithm(searchAlgoritm);
	    this.fN=0;
	    this.fP=0;
	    this.tN=0;
	    this.tP=0;
	}
	
	public DistanceType getDistance() {
		return distance;
	}

	public void setDistance(DistanceType distance) {
		this.distance = distance;
		switch (distance) {
		case Manhattan:
			this.distanceMethod = new Minkowski(1);
			break;
		case Euclidea:
			this.distanceMethod = new Minkowski(2);
			break;
		case Minkowski:
			this.distanceMethod = new Minkowski(3);
			break;
		default:
			break;
		}
	}

	public void setDistanceWeighting(DistanceWight i){
		this.distanceWeighting=i;
		
	}
	public DistanceWight getDistanceWeighting(){
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
		double distancia = 0.00;
		lista = new double[instancias.numInstances()][2];
		
		//primera linea referencia instancia; segunda linea distancia
		for(int j=1;j<=instancias.numInstances();j++){
			distancia=this.calcularDistancia(instancias.get(j),instancia);
			lista[j][0]=j;
			lista[j][1]=distancia;
		}
		Collections.sort(Arrays.asList(lista), new Comparator<double[]>() {

			@Override
			public int compare(double[] o1, double[] o2) {
				return Double.compare(o1[1], o2[1]);
			}
		});
	}
	public void crearMatrixConfusion(Instances noclasf, Instances clasf){
		// 1 sera t 0 sera f
		for (int i = 0; i < clasf.numInstances(); i++) {
			double clase = clasf.get(i).classValue();
			System.out.println(clase);
			if(clasf.get(i).classValue()==noclasf.get(i).classValue() &&  clase== 1.0){
				tP=tP++;
			}else if(clasf.get(i).classValue()!=noclasf.get(i).classValue() && clase== 1.0){
				tN=tN++;
			}else if(clasf.get(i).classValue()==noclasf.get(i).classValue() && clase== 0.0){
				fP=fP++;
			}else if(clasf.get(i).classValue()!=noclasf.get(i).classValue() && clase== 0.0){
				fN=fN++;
			}
		}
		
		System.out.println("Matr�z realizada con �xito");
	}
	
	public void calcularMediciones(){
		float recall;
		float accuracy;
		float precision;
		float fmeasure;
		
		precision=100*(this.tP/(this.tP + this.fP));
		recall=this.tP/(this.tP + this.fN);
		accuracy=(this.tP + this.tN) / (this.tP + this.tN + this.fP + this.fN);
		fmeasure=(2*precision*recall)/(precision + recall); 
		Escritor.getEscritor().hacerFicheroNuestroModelo("ficheros/EvaluationNuestroModelo.txt",this.getKNN(),this.getDistanceWeighting(), this.getNearestNeighbourSearchAlgorithm(), precision, recall, accuracy, fmeasure, tP, tN, fP, fN , true);
	}
	private double calcularDistancia(Instance instancia,Instance instanciaaclasificar) {
		int numAtr = instanciaaclasificar.numAttributes();
		for(int i=1;i<=numAtr;i++){
			this.distanceMethod.setAtributeDist(instancia.value(i), instanciaaclasificar.value(i));
		}
		return this.distanceMethod.getDistance();
	}

	public double clasificarInstancia(Instance NoClasificada,Instances instancias){
		int numerovecinos = this.getKNN();
		int metodo = this.getNearestNeighbourSearchAlgorithm();
		//recorreremos el array hasta el numero de k en instancias
		if (metodo == 1){
			ArrayList<Instance> instanciasOrdenadas = new ArrayList<Instance>();
			ArrayList<Integer> distancias = new ArrayList<Integer>();

			for(int i=1;i<=numerovecinos;i++){
				instanciasOrdenadas.add(instancias.get(lista[i][0]));
				distancias.add(instancias.get((int)lista[i][1]));
				
				
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
		//devolveremos las intancias clasificadas : solaparemos las clases con la calculada en el metodo. 
		return 0.00;
	}
}	
