package src;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.rules.OneR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.neighboursearch.BallTree;
import weka.core.neighboursearch.CoverTree;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class Modelo {

	
	
	public static void main(String[] args){
		
		Evaluation evaluator = null;
		
		//trainaurre eta devaurre entrenamendu multzoak jasoko dira. Lehenik beti jarriko da train multzoa eta gero dev multzoa
		Instances preTrain = Lector.getLector().leerInstancias(args[0]);
		Instances preDev = Lector.getLector().leerInstancias(args[1]);
		
		// IBk

		IBk estimador= new IBk();
		
		//Distancias
		ArrayList<SelectedTag> distances= new ArrayList<SelectedTag>();
		//NeighbourSearch	
		ArrayList<NearestNeighbourSearch> neighbours= new ArrayList<NearestNeighbourSearch>();
	               
		System.out.println("Trabajando...");
		prepararArray(estimador, distances,neighbours);
		
		double fmeasureMedia=0;
		double fmeasureMediaMax=0;            
	    int metodoNeighbourmax=0; 
	    int distancemax=0;
	    int kMax=0;
	                	
		for (int j=1;j<=preTrain.numInstances();j++){   //K-NN  loop
	        estimador.setKNN(j);
			//NeighbourSearchAlgorithm loop
			for (int z=0; z<neighbours.size();z++){
				estimador.setNearestNeighbourSearchAlgorithm(neighbours.get(z));
				// Distance desberdinen loop-a
				for (int x=0;x<distances.size();x++){
					estimador.setDistanceWeighting(distances.get(x));	
					try {
						estimador.buildClassifier(preTrain);
						//Inicializar evaluador
						evaluator = new Evaluation(preTrain);
				
						evaluator.evaluateModel(estimador, preDev);
						//klase minoritariaren f-measurearekin konparatuz.
						fmeasureMedia = evaluator.weightedFMeasure();
						if(fmeasureMedia>fmeasureMediaMax){
							kMax=j;
							metodoNeighbourmax=z;
							distancemax=x;
							fmeasureMediaMax=fmeasureMedia;
						}
					}catch(Exception e){
						e.printStackTrace(); System.exit(1);
						
						
					}
				}
			}
	       }
	        
		// hacer busqueda no exhaustiba
		
		CVParameterSelection BusquedaNoExhaustiba = new CVParameterSelection();
		BusquedaNoExhaustiba.setClassifier(estimador);
		try {
			BusquedaNoExhaustiba.setNumFolds(5);
			BusquedaNoExhaustiba.addCVParameter("K 1 "+preTrain.numInstances()+" " +preTrain.numInstances());
			BusquedaNoExhaustiba.buildClassifier(preTrain);
		} catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
		IBk clasificador = (IBk)BusquedaNoExhaustiba.getClassifier();
	    int knnEzExhaustiboa= clasificador.getKNN();
	    
	    // Inferencia
	    
	    estimador.setDistanceWeighting(distances.get(distancemax));
	    estimador.setKNN(kMax);
	    estimador.setNearestNeighbourSearchAlgorithm(neighbours.get(metodoNeighbourmax));
	    
	    // Hold out 70 30
	    try{
	    	evaluator = new Evaluation(preTrain);
	    	estimador.buildClassifier(preTrain);
	    	evaluator.evaluateModel(estimador, preDev);
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    Escritor.getEscritor().hacerFicheroIBk("ficheros/EvaluationIBk.txt", evaluator, estimador, knnEzExhaustiboa, "Hold Out 70 30",true);
	    System.out.println("70 30 base");
	   
	    
	    //modeloa egin
	    Escritor.getEscritor().escribirModelo("modelos/IBkModel.model", estimador);
	    
	    System.out.println("Modelo IBk escrito");  
		
	}
	
	private static int minorityclassindex(Instances i){
		int kont [] = new int [i.numClasses()];
		for (int j : kont) {
			kont[j] = 0;
		}
		for (Instance instance : i) {
			kont[1+(instance.classAttribute().indexOfValue(""+instance.classValue()))] +=1;
		}
		int min = kont[0];
		int ind = 0;
		for (int j = 1; j < kont.length; j++)  {
			if(kont[j] < min){
				min = kont[j];
				ind = j;
			}
		}
		return ind;
	}
	
	private static void prepararArray(IBk estimador, ArrayList<SelectedTag> distances, ArrayList<NearestNeighbourSearch> neighbours) {
		
		//Distance
		for(int a=1;a<IBk.TAGS_WEIGHTING.length;a++){
			SelectedTag s= new SelectedTag(a,IBk.TAGS_WEIGHTING);
			distances.add(s);
		}
		//Neighbour
		BallTree balltree= new BallTree();
		KDTree kdtree= new KDTree();
		LinearNNSearch linearNNSearch= new LinearNNSearch();
		CoverTree covertree= new CoverTree();
                
                
		neighbours.add(balltree);
		neighbours.add(kdtree);
		neighbours.add(linearNNSearch);
		neighbours.add(covertree);
    }
	
}
