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
		
		//klase minoritaria kalkulatu
		//int min = minorityclassindex(trainaurre);
		
		// train eta dev gehitu
	    Instances trainydev=new Instances(preTrain);
	    trainydev.addAll(preDev);
		
		//train eta dev-ren aldaerak eraiki.
		int trainSize = (int) Math.round(trainydev.numInstances() * 0.7);
    	int testSize = trainydev.numInstances() - trainSize;
	    Instances trainydev70 = new Instances(trainydev, 0, trainSize);
    	Instances trainydev30 = new Instances(trainydev, trainSize, testSize);
		
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
	    
	    // No honesta
	    try {
	    	evaluator = new Evaluation(trainydev);
			estimador.buildClassifier(trainydev);
			evaluator.evaluateModel(estimador, trainydev);
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	   
	    Escritor.getEscritor().hacerFicheroIBk("ficheros/EvaluationIBk.txt", evaluator, estimador, knnEzExhaustiboa, "No honesta",false);
	    
	    // Hold out 70 30
	    try{
	    	evaluator = new Evaluation(trainydev70);
	    	estimador.buildClassifier(trainydev70);
	    	evaluator.evaluateModel(estimador, trainydev30);
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    Escritor.getEscritor().hacerFicheroIBk("ficheros/EvaluationIBk.txt", evaluator, estimador, knnEzExhaustiboa, "Hold Out 70 30",true);
	    System.out.println("70 30 base");
	    // hold out train dev
	    try{
	    	evaluator = new Evaluation(preTrain);
	    	estimador.buildClassifier(preTrain);
	    	evaluator.evaluateModel(estimador, preDev);
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    System.out.println("Train eta dev base");
	    
	    Escritor.getEscritor().hacerFicheroIBk("ficheros/EvaluationIBk.txt", evaluator, estimador, knnEzExhaustiboa, "Hold Out train dev",true);
 
	    // 10 Fold cross validation
	    try{
	    	evaluator = new Evaluation(trainydev);
	    	estimador.buildClassifier(trainydev);
	    	evaluator.crossValidateModel(estimador, trainydev, 10, new Random(1));
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    
	    Escritor.getEscritor().hacerFicheroIBk("ficheros/EvaluationIBk.txt", evaluator, estimador, knnEzExhaustiboa, "10 Fold cross",true);
	    System.out.println("10 fold x validation base");
	    //modeloa egin
	    Escritor.getEscritor().escribirModelo("modelos/IBkModel.model", estimador);
	    
	    System.out.println("Modelo IBk escrito");
		
		// NuestroModelo
		NuestroModelo estimadorNuestro = new NuestroModelo();
		
		
		double fmeasureMediaMulti=0;
		double fmeasureMediaMaxMulti=0;
		double rateMax = 0.0;
		double momentumMax = 0.2;
		int trainingtimemax = 0;
		boolean decaymax = false;
		
		//HIDDEN LAYERS AUKERAK A, I , O , T DIRENEZ, array batean gorde
		ArrayList<String> hiddenlayers= new ArrayList<String>();
		hiddenlayers.add("a");
		hiddenlayers.add("i");
		hiddenlayers.add("o");
		hiddenlayers.add("t");
        String hiddenlayersMax= "";
        estimadorMulti.setAutoBuild(true); // hidden layers
        
        //Aurreprozesatzailean egiten direlako.
        estimadorMulti.setNominalToBinaryFilter(false);
        estimadorMulti.setNormalizeAttributes(false);
        estimadorMulti.setNormalizeNumericClass(false);
        
        //estimadorMulti.setGUI(true);
		for (int i=0;i<hiddenlayers.size();i++){
			//hidden layer egokiena aukeratzeko loopa, f-measure altuenaren bila.
			estimadorMulti.setHiddenLayers(hiddenlayers.get(i));
			for (double rate = 0.2; rate<=1; rate+=0.2){
				estimadorMulti.setLearningRate(rate);
				for(double momentum = 0.2; momentum<=1; momentum+=0.2){
					estimadorMulti.setMomentum(momentum);
					for (int trainingtime = 5; trainingtime < 50; trainingtime+=5){
						estimadorMulti.setTrainingTime(trainingtime);
						for (int decay = 0; decay < 2; decay++) {
							estimadorMulti.setDecay(decay<1);
							System.out.println("##############\nhiddenlayers:"+i+"\nrate:"+rate+"\nmomentum:"+momentum+"\ntrainigtime:"+trainingtime+"\nDecay:"+decay);
							try{
								evaluator = new Evaluation(trainaurre);
								estimadorMulti.buildClassifier(trainaurre);
								evaluator.evaluateModel(estimadorMulti, devaurre);
								// klase minoritariaren fmeasurearekin konparatu
								fmeasureMediaMulti = evaluator.fMeasure(minorityclassindex(trainetadev));
								evaluator.errorRate();
								if(evaluator.errorRate()<errorratemax){
									errorratemax = evaluator.errorRate();
									hiddenlayersMax =  hiddenlayers.get(i);
									rateMax = rate;
									momentumMax=momentum;
									trainingtimemax= trainingtime;
									decaymax=(decay<1);
									
								}
								if(fmeasureMediaMulti>fmeasureMediaMaxMulti){
									fmeasureMediaMaxMulti = fmeasureMediaMulti;
									hiddenlayersMax =  hiddenlayers.get(i);
									rateMax = rate;
									momentumMax=momentum;
									trainingtimemax= trainingtime;
									decaymax=(decay<1);
									
								}
							} catch (Exception e) {
								e.printStackTrace(); System.exit(1);
							}
						}
					}
				}
			}
		}
		CVParameterSelection bilaketaEzExhaustiboaNuestro = new CVParameterSelection();
		bilaketaEzExhaustiboaNuestro.setClassifier((Classifier) estimadorNuestro);
		try{
			bilaketaEzExhaustiboaNuestro.buildClassifier(preTrain);
		} catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
		MultilayerPerceptron clasiNuestro = (MultilayerPerceptron)bilaketaEzExhaustiboaNuestro.getClassifier();
	    String hiddenLayerEzexhaustiboa= clasiNuestro.getHiddenLayers();
	    System.out.println("MultiLayer Perceptron prozesuaren emaitzak imprimatzen:");
	
	    //Inferentzia
	    
	    estimadorNuestro.setDistanceWeighting();
		estimadorNuestro.setNearestNeighbourSearchAlgorithm();
		estimadorNuestro.setKNN();
	    
	    // No Honesta
	    try{
	    	evaluator = new Evaluation(trainydev);
	    	estimadorNuestro.buildClassifier(trainydev);
	    	evaluator.evaluateModel(estimadorNuestro, trainydev);
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    Escritor.getEscritor().hacerFicheroNuestroModelo("ficheros/EvaluationNuestroModelo.txt", evaluator, estimadorNuestro, hiddenLayerEzexhaustiboa, "No honesta",false);
	    System.out.println("No Fair Mp");
	    // Hold out 70 30
	    try{
	    	evaluator = new Evaluation(trainydev70);
	    	estimadorNuestro.buildClassifier(trainydev70);
	    	evaluator.evaluateModel(estimadorNuestro, trainydev);
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    
	    Escritor.getEscritor().hacerFicheroNuestroModelo("ficheros/EvaluationNuestroModelo.txt", evaluator, estimadorNuestro, hiddenLayerEzexhaustiboa, "Hold Out 70 30",true);	    
	    System.out.println("70 30 MP");
	    // hold out train dev
	    try{
	    	evaluator = new Evaluation(preTrain);
	    	estimadorNuestro.buildClassifier(preTrain);
	    	evaluator.evaluateModel(estimadorNuestro, preDev);
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    Escritor.getEscritor().hacerFicheroNuestroModelo("ficheros/EvaluationNuestroModelo.txt", evaluator, estimadorNuestro, hiddenLayerEzexhaustiboa, "Hold Out train dev",true);	    
	    System.out.println("HoldOut MP");
	    // 10 Fold cross validation
	    try{
	    	evaluator = new Evaluation(trainydev);
	    	estimadorNuestro.buildClassifier(trainydev);
	    	evaluator.crossValidateModel(estimadorNuestro, trainydev, 10, new Random(1));
	    } catch (Exception e) {
			e.printStackTrace(); System.exit(1);
		}
	    System.out.println("10 fold X validation MP");
	    
	    Escritor.getEscritor().hacerFicheroNuestroModelo("ficheros/EvaluationMultilayerPerceptron.txt", evaluator, estimadorNuestro, hiddenLayerEzexhaustiboa, "10 Fold cross",true);
		
	    // Hacer el Modelo
	    Escritor.getEscritor().escribirModelo("modelos/NuestroModeloModel.model", estimadorNuestro);
	    System.out.println("Escrito nuestro modelo");
	    
	    
		
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
