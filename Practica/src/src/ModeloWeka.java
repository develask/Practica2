package src;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.MinkowskiDistance;
import weka.core.SelectedTag;
import weka.core.neighboursearch.LinearNNSearch;

public class ModeloWeka implements Modelo {
	
	private IBk estimador;
	private SelectedTag w;
	private Evaluation evaluator;
	private int k,t;
	public ModeloWeka(int k, int w, int t){
		this.k = k;
		this.t = t;
		try {
			estimador= new IBk();
			estimador.setKNN(k);
			LinearNNSearch search = new LinearNNSearch();
			MinkowskiDistance minkowski = new MinkowskiDistance();
			minkowski.setOrder(t);
			search.setDistanceFunction(minkowski);
			estimador.setNearestNeighbourSearchAlgorithm(search);
			this.w = new SelectedTag(w,IBk.TAGS_WEIGHTING);
			estimador.setDistanceWeighting(this.w);
		} catch (Exception e) {
			System.out.println("Ha ocurrido algún error.");
			e.printStackTrace();
		}
	}
	
	


	@Override
	public String calcularMediciones() {
		try {
			String result = "";
			result += ("\n**************************************\n");
			result += ("\n****Estimacion modelo k-NN de WEKA****\n");
			result += ("\n**************************************\n");
			result += ("K: " + this.k+"\n");
			result += ("Distance Weighting: " + this.w+"\n");
			result += ("Distance Type (Minkowski): " + this.t+"\n");
			result+= this.evaluator.toSummaryString();
			result += this.evaluator.toMatrixString();
			return result;
		} catch (Exception e) {
			System.out.println("Hay algún error con el evaluador.");
			e.printStackTrace();
			return "";
		}
	}


	@Override
	public double accuracy() {
		return evaluator.unweightedMicroFmeasure();
	}


	@Override
	public void buildClasifier(Instances instancias) {
		try {
			this.evaluator = new Evaluation(instancias);
			this.estimador.buildClassifier(instancias);
		} catch (Exception e) {
			System.out.println("Hay algún problema con el estimador");
			e.printStackTrace();
		}
		
	}


	@Override
	public void evaluarModelo(Instances instanciasAEvaluar) {
		try {
			evaluator.evaluateModel(estimador, instanciasAEvaluar);
		} catch (Exception e) {
			System.out.println("Ha habido algún problema al evaluar las instancias.");
			e.printStackTrace();
		}
	}
	
}
