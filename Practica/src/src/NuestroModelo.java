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
		NoDistance, OneDivDistance
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
	
	
	private double tP;
	private double tN;
	private double fP;
	private double fN;
	
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
	    this.fN=0.001;
	    this.fP=0.001;
	    this.tN=0.001;
	    this.tP=0.001;
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
		for(int j=0;j<instancias.numInstances();j++){
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
			double clase = noclasf.get(i).classValue();
			//System.out.println(clase+"-"+clasf.get(i).classValue());
			//System.out.println(clasf.classAttribute().value((int)clase));
			if(clasf.get(i).classValue()==noclasf.get(i).classValue() &&  clase== 1.0){
				tP++;
			}else if(clasf.get(i).classValue()!=noclasf.get(i).classValue() && clase== 1.0){
				tN++;
			}else if(clasf.get(i).classValue()==noclasf.get(i).classValue() && clase== 0.0){
				fP++;
			}else if(clasf.get(i).classValue()!=noclasf.get(i).classValue() && clase== 0.0){
				fN++;
			}
		}
		
		System.out.println("Matríz realizada con Éxito");
	}
	
	public double calcularMediciones(double fm){
		double recall;
		double accuracy;
		double precision;
		double fmeasure;
		
		precision=100*(this.tP/(this.tP + this.fP));
		recall=100*(this.tP/(this.tP + this.fN));
		accuracy=(this.tP + this.tN) / (this.tP + this.tN + this.fP + this.fN);
		fmeasure=(2*precision*recall)/(precision + recall); 
		if (fm < accuracy){
			Escritor.getEscritor().hacerFicheroNuestroModelo("ficheros/EvaluationNuestroModelo.txt",this.getKNN(),this.getDistanceWeighting(),this.getDistance(), this.getNearestNeighbourSearchAlgorithm(), precision, recall, accuracy, fmeasure,(int)Math.floor(tP),(int)Math.floor(tN),(int)Math.floor(fP),(int)Math.floor(fN) , false);
			return accuracy;
		}else{
			return fm;
		}
	}
	private double calcularPeso(double distancia){
		
		switch (this.getDistanceWeighting()) {
		case NoDistance:
			return distancia;
		case OneDivDistance:
			return 1/distancia;
		}
		return distancia;
	}
	private double calcularDistancia(Instance instancia,Instance instanciaaclasificar) {
		this.distanceMethod.init();
		int numAtr = instanciaaclasificar.numAttributes();
		for(int i=0;i<numAtr;i++){
			this.distanceMethod.setAtributeDist((double)instancia.value(i), (double)instanciaaclasificar.value(i));
		}
		return this.distanceMethod.getDistance();
	}

	public double clasificarInstancia(Instance NoClasificada,Instances instancias){
		int numerovecinos = this.getKNN();
		int metodo = this.getNearestNeighbourSearchAlgorithm();
		//recorreremos el array hasta el numero de k en instancias
		if (metodo == 1){
			double[] mediasPeso = new double[instancias.numClasses()];
			Double[][] temp= new Double[instancias.numClasses()][2];
			for(int i=0;i<temp.length;i++){
				for (int j = 0; j < temp[i].length; j++) {
					temp[i][j]=0.00;
				}
			}
			for(int i=0;i<numerovecinos;i++){
				temp[(int)instancias.get((int)lista[i][0]).classValue()][0]+=calcularPeso(lista[i][1]);
				temp[(int)instancias.get((int)lista[i][0]).classValue()][1]++;
			}
			for(int i=0;i<instancias.numClasses();i++){
				if(temp[i][1]>0){
					mediasPeso[i]=temp[i][0]/temp[i][1];
				}else{
					mediasPeso[i]=-1;
				}
			}
			return conseguirClase(mediasPeso);//==0.0?1.0:0.0;
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

	private double conseguirClase(double[] mediasPeso) {
		int pos=-1;
		double peso=Double.MAX_VALUE;
		switch (this.getDistanceWeighting()) {
		case NoDistance:
			for(int i=0;i<mediasPeso.length;i++){					
				if(mediasPeso[i]<peso && mediasPeso[i]!=-1){
					pos = i;
					peso = mediasPeso[i];
				}
			}
			break;
		case OneDivDistance:
			peso=Double.MIN_VALUE;
			for(int i=0;i<mediasPeso.length;i++){					
				if(mediasPeso[i]>peso && mediasPeso[i]!=-1){
					pos = i;
					peso = mediasPeso[i];
				}
			}
			break;
		}
		return pos;
		
	}
}	
