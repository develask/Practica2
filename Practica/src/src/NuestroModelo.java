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
import weka.core.neighboursearch.NearestNeighbourSearch;
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

import java.util.Enumeration;
import java.util.Vector;

public class NuestroModelo extends AbstractClassifier implements OptionHandler, UpdateableClassifier, WeightedInstancesHandler, TechnicalInformationHandler, AdditionalMeasureProducer { 
	
	   static final long serialVersionUID = -3080186098777067172L;
	   protected Instances Train;
	   protected int NumClasses;
	   protected int ClassType;
	   protected int kNN;
	   protected int kNNMaximo;
	   protected boolean kNNvalido;
	   protected int WindowSize;
	   protected int DistanceWeighting;
	   protected boolean CrossValidate;
	   protected boolean MeanSquared;
	   protected ZeroR defaultModel;
	   public static final int WEIGHT_NONE = 1;
	   public static final int WEIGHT_INVERSE = 2;
	   public static final int WEIGHT_SIMILARITY = 4;
	   public static final Tag [] TAGS_WEIGHTING = {
		    new Tag(WEIGHT_NONE, "No distance weighting"),
		    new Tag(WEIGHT_INVERSE, "Weight by 1/distance"),
		    new Tag(WEIGHT_SIMILARITY, "Weight by 1-distance")
	   };
	   protected NearestNeighbourSearch NNSearch = new LinearNNSearch();
	   protected double numAttributesUsed;
	
	public NuestroModelo(int k){
		init();
	    setKNN(k);
	}
	  
	public NuestroModelo() { 
	     init();
	}

	public int getKNN(){
		return kNN;
	}
	
	public void setDistanceWeighting(int distance) {
		DistanceWeighting=distance;
	}
	  
	public int getWindowSize() {
		return WindowSize;
	} 
	  
	public void setWindowSize(int newWindowSize) {     
	     WindowSize = newWindowSize;
	}

	public void setKNN(int k) {
		this.kNN=k;
		this.kNNMaximo=k;
		this.kNNvalido=false;
	}
	public SelectedTag getDistanceWeighting() {
		return new SelectedTag(DistanceWeighting, TAGS_WEIGHTING);
	}
	
	public void setDistanceWeighting(SelectedTag newMethod) {   
		if (newMethod.getTags() == TAGS_WEIGHTING) {
			DistanceWeighting = newMethod.getSelectedTag().getID();
		}
	}
	public boolean getMeanSquared() {    
		return MeanSquared;
	}
	  
	public void setMeanSquared(boolean newMeanSquared) {     
		MeanSquared = newMeanSquared;
	}
	public boolean getCrossValidate() {     
		return CrossValidate;
	}
	public void setCrossValidate(boolean newCrossValidate) {     
		CrossValidate = newCrossValidate;
	}
	
	public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
		return NNSearch;
	}
	
	public void setNearestNeighbourSearchAlgorithm(NearestNeighbourSearch nearestNeighbourSearchAlgorithm) {
		NNSearch = nearestNeighbourSearchAlgorithm;
	}
	
	   
	public int getNumTraining() {
		return Train.numInstances();
	}
	
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		
		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		
		// instances
		result.setMinimumNumberInstances(0);
		    
		return result;
	}
	  
	  
	
	public void buildClassifier(Instances instances) throws Exception {
	   
	    // can classifier handle the data?
	    getCapabilities().testWithFail(instances);
	
	    // remove instances with missing class
	    instances = new Instances(instances);
	    instances.deleteWithMissingClass();
	   
	    NumClasses = instances.numClasses();
	    ClassType = instances.classAttribute().type();
	    Train = new Instances(instances, 0, instances.numInstances());
	 
	    // Throw away initial instances until within the specified window size
	    if ((WindowSize > 0) && (instances.numInstances() > WindowSize)) {
	      Train = new Instances(Train, Train.numInstances()-WindowSize, WindowSize);
	    }
	
	    numAttributesUsed = 0.0;
	    for (int i = 0; i < Train.numAttributes(); i++) {
	      if ((i != Train.classIndex()) && 
		  (Train.attribute(i).isNominal() ||
		   Train.attribute(i).isNumeric())) {
	    	  numAttributesUsed += 1.0;
	      }
		}     
	    NNSearch.setInstances(Train);
	 
	    // Invalidate any currently cross-validation selected k
	    kNNvalido = false;
	     
	    defaultModel = new ZeroR();
	    defaultModel.buildClassifier(instances);
	}

	@Override
	public void updateClassifier(Instance arg0) throws Exception {
	 //cambiar; pasar a codigo el pseudocogido, y evaluarlo en el build classifier desde
		// nuestro algoritmo y no desde el IBk
		
		
		if (Train.equalHeaders(arg0.dataset()) == false) {
	       throw new Exception("Incompatible instance types\n" + Train.equalHeadersMsg(arg0.dataset()));
	     }
	     if (arg0.classIsMissing()) {
	       return;
	     }
	     Train.add(arg0);
	     NNSearch.update(arg0);
	     kNNvalido = false;
	     if ((WindowSize > 0) && (Train.numInstances() >WindowSize)) {
	       boolean deletedInstance=false;
	       while (Train.numInstances() > WindowSize) {
	    	   Train.delete(0);
	    	   deletedInstance=true;
	       }
	       //rebuild datastructure KDTree currently can't delete
	       if(deletedInstance==true)
	    	   NNSearch.setInstances(Train);
	     }
   	}
	public double [] distributionForInstance(Instance instance) throws Exception {
		    if (Train.numInstances() == 0) {
		      //throw new Exception("No training instances!");
		       return defaultModel.distributionForInstance(instance);
		    }
		    if ((WindowSize > 0) && (Train.numInstances() > WindowSize)) {
		       kNNvalido = false;
		      boolean deletedInstance=false;
		      while (Train.numInstances() > WindowSize) {
		 	Train.delete(0);
		       }
		       //rebuild datastructure KDTree currently can't delete
		       if(deletedInstance==true)
		         NNSearch.setInstances(Train);
		     }
		 
		     // Select k by cross validation
		     if (!kNNvalido && (CrossValidate) && (kNNMaximo >= 1)) {
		       crossValidate();
		     }
		 
		     NNSearch.addInstanceInfo(instance);
		 
		     Instances neighbours = NNSearch.kNearestNeighbours(instance, kNN);
		     double [] distances = NNSearch.getDistances();
		     double [] distribution = makeDistribution( neighbours, distances );
		 
		     return distribution;
   }
	

	  
	

	  
   public Enumeration listOptions() {
 
	   	Vector newVector = new Vector(8);
	 
	    newVector.addElement(new Option(
	 	      "\tWeight neighbours by the inverse of their distance\n"+
	 	      "\t(use when k > 1)",
	 	      "I", 0, "-I"));
	    newVector.addElement(new Option(
	 	      "\tWeight neighbours by 1 - their distance\n"+
	 	      "\t(use when k > 1)",
	 	      "F", 0, "-F"));
	    newVector.addElement(new Option(
	 	      "\tNumber of nearest neighbours (k) used in classification.\n"+
	 	      "\t(Default = 1)",
	 	      "K", 1,"-K <number of neighbors>"));
	    newVector.addElement(new Option(
	           "\tMinimise mean squared error rather than mean absolute\n"+
	 	      "\terror when using -X option with numeric prediction.",
	 	      "E", 0,"-E"));
	    newVector.addElement(new Option(
	           "\tMaximum number of training instances maintained.\n"+
	 	      "\tTraining instances are dropped FIFO. (Default = no window)",
	 	      "W", 1,"-W <window size>"));
	    newVector.addElement(new Option(
	 	      "\tSelect the number of nearest neighbours between 1\n"+
	 	      "\tand the k value specified using hold-one-out evaluation\n"+
	 	      "\ton the training data (use when k > 1)",
	 	      "X", 0,"-X"));
	    newVector.addElement(new Option(
	 	      "\tThe nearest neighbour search algorithm to use "+
	           "(default: weka.core.neighboursearch.LinearNNSearch).\n",
	 	      "A", 0, "-A"));
	 
	    return newVector.elements();
   }
   public void setOptions(String[] options) throws Exception {
	    
	     String knnString = Utils.getOption('K', options);
	     if (knnString.length() != 0) {
	       setKNN(Integer.parseInt(knnString));
	     } else {
	       setKNN(1);
	     }
	     String windowString = Utils.getOption('W', options);
	     if (windowString.length() != 0) {
	       setWindowSize(Integer.parseInt(windowString));
	     } else {
	       setWindowSize(0);
	     }
	     if (Utils.getFlag('I', options)) {
	       setDistanceWeighting(new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING));
	     } else if (Utils.getFlag('F', options)) {
	       setDistanceWeighting(new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING));
	     } else {
	       setDistanceWeighting(new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING));
	     }
	     setCrossValidate(Utils.getFlag('X', options));
	     setMeanSquared(Utils.getFlag('E', options));
	 
	     String nnSearchClass = Utils.getOption('A', options);
	     if(nnSearchClass.length() != 0) {
	    	String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
      		if(nnSearchClassSpec.length == 0) { 
      			throw new Exception("Invalid NearestNeighbourSearch algorithm " +
                             "specification string."); 
      		}
	        String className = nnSearchClassSpec[0];
	        nnSearchClassSpec[0] = "";
	 
	        setNearestNeighbourSearchAlgorithm( (NearestNeighbourSearch)
	        Utils.forName( NearestNeighbourSearch.class, className, nnSearchClassSpec));
	     }
	     else 
	       this.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());
	     
	     Utils.checkForRemainingOptions(options);
   }
   
   public String [] getOptions() {
		 
	     String [] options = new String [11];
	     int current = 0;
	     options[current++] = "-K"; options[current++] = "" + getKNN();
	     options[current++] = "-W"; options[current++] = "" + WindowSize;
	     if (getCrossValidate()) {
	       options[current++] = "-X";
	     }
	     if (getMeanSquared()) {
	       options[current++] = "-E";
	     }
	     if (DistanceWeighting == WEIGHT_INVERSE) {
	       options[current++] = "-I";
	     } else if (DistanceWeighting == WEIGHT_SIMILARITY) {
	       options[current++] = "-F";
	     }
	 
	     options[current++] = "-A";
	     options[current++] = NNSearch.getClass().getName()+" "+Utils.joinOptions(NNSearch.getOptions()); 
	     
	     while (current < options.length) {
	       options[current++] = "";
	     }
		     
		     return options;
   }
   
   public Enumeration enumerateMeasures() {
		     if (CrossValidate) {
		       Enumeration enm = NNSearch.enumerateMeasures();
		       Vector measures = new Vector();
		       while (enm.hasMoreElements())
		 	measures.add(enm.nextElement());
		       measures.add("measureKNN");
		       return measures.elements();
		     }
		     else {
		       return NNSearch.enumerateMeasures();
		     }
   }
   
   public double getMeasure(String additionalMeasureName) {
	   	 if (additionalMeasureName.equals("measureKNN"))
	       return kNN;
	     else
	       return NNSearch.getMeasure(additionalMeasureName);
   }
   
   public String toString() {
		 
		     if (Train == null) {
		       return "IBk: No model built yet.";
		     }
		     
		     if (Train.numInstances() == 0) {
		       return "Warning: no training instances - ZeroR model used.";
		     }
		 
		     if (!kNNvalido && CrossValidate) {
		       crossValidate();
		     }
		     
		     String result = "IB1 instance-based classifier\n" +
		       "using " + kNN;
		 
		     switch (DistanceWeighting) {
		     case WEIGHT_INVERSE:
		       result += " inverse-distance-weighted";
		       break;
		     case WEIGHT_SIMILARITY:
		       result += " similarity-weighted";
		       break;
		     }
		     result += " nearest neighbour(s) for classification\n";
		 
		     if (WindowSize != 0) {
		       result += "using a maximum of " 
		 	+ WindowSize + " (windowed) training instances\n";
		     }
		     return result;
		   }
		 
   
   	protected void init() { 
	     setKNN(1);
	     WindowSize = 0;
	     DistanceWeighting = WEIGHT_NONE;
	     CrossValidate = false;
	     MeanSquared = false;
   	}
   	
   	public String meanSquaredTipText() {
   	    return "Whether the mean squared error is used rather than mean "
   	       + "absolute error when doing cross-validation for regression problems.";
   	}	  
   	
   	public String crossValidateTipText() {
        return "Whether hold-one-out cross-validation will be used " + "to select the best k value.";
   	}
   	
 	public String nearestNeighbourSearchAlgorithmTipText() {
	     return "The nearest neighbour search algorithm to use " + "(Default: weka.core.neighboursearch.LinearNNSearch).";
	}
 	
	public String distanceWeightingTipText() {
	     return "Gets the distance weighting method used.";
	}	

   	
   	public Instances pruneToK(Instances neighbours, double[] distances, int k) {    
	    if(neighbours==null || distances==null || neighbours.numInstances()==0) {
	      return null;
	    }
	    if (k < 1) {
	      k = 1;
	    }
	    
	    int currentK = 0;
	    double currentDist;
	    for(int i=0; i < neighbours.numInstances(); i++) {
	      currentK++;
	      currentDist = distances[i];
	      if(currentK>k && currentDist!=distances[i-1]) {
	        currentK--;
	        neighbours = new Instances(neighbours, 0, currentK);
	        break;
	      }
	    }
	
	   return neighbours;
	}
   	protected double [] makeDistribution(Instances neighbours, double[] distances) throws Exception {
	     double total = 0, weight;
	     double [] distribution = new double [NumClasses];
	     
	     // Set up a correction to the estimator
	     if (ClassType == Attribute.NOMINAL) {
	       for(int i = 0; i < NumClasses; i++) {
	 	distribution[i] = 1.0 / Math.max(1,Train.numInstances());
	       }
	       total = (double)NumClasses / Math.max(1,Train.numInstances());
	     }
	 
	     for(int i=0; i < neighbours.numInstances(); i++) {
	       // Collect class counts
	       Instance current = neighbours.instance(i);
	       distances[i] = distances[i]*distances[i];
	       distances[i] = Math.sqrt(distances[i]/numAttributesUsed);
	       switch (DistanceWeighting) {
	         case WEIGHT_INVERSE:
	           weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
	           break;
	         case WEIGHT_SIMILARITY:
	           weight = 1.0 - distances[i];
	           break;
	         default:                                 // WEIGHT_NONE:
	           weight = 1.0;
	           break;
	       }
	       weight *= current.weight();
	       try {
	         switch (ClassType) {
	           case Attribute.NOMINAL:
	             distribution[(int)current.classValue()] += weight;
	             break;
	           case Attribute.NUMERIC:
	            distribution[0] += current.classValue() * weight;
	             break;
	         }
	       } catch (Exception ex) {
	         throw new Error("Data has no class attribute!");
	       }
	       total += weight;      
	     }
	 
	     // Normalise distribution
	     if (total > 0) {
	       Utils.normalize(distribution, total);
	     }
	     return distribution;
	   }
  
   protected void crossValidate() {
     try {
	       if (NNSearch instanceof weka.core.neighboursearch.CoverTree) throw new Exception("CoverTree doesn't support hold-one-out "+ "cross-validation. Use some other NN " + "method.");
	 
	       double [] performanceStats = new double [kNNMaximo];
	       double [] performanceStatsSq = new double [kNNMaximo];
	 
	       for(int i = 0; i < kNNMaximo; i++) {
	    	   performanceStats[i] = 0;
	    	   performanceStatsSq[i] = 0;
	       }
	 
	 
		    kNN = kNNMaximo;
		    Instance instance;
		    Instances neighbours;
		    double[] origDistances, convertedDistances;
		    for(int i = 0; i < Train.numInstances(); i++) {
			 	if (m_Debug && (i % 50 == 0)) {
			 	  System.err.print("Cross validating "
			 			   + i + "/" + Train.numInstances() + "\r");
			 	}
			 	instance = Train.instance(i);
			 	neighbours = NNSearch.kNearestNeighbours(instance, kNN);
			 	origDistances = NNSearch.getDistances();
		         
			 	for(int j = kNNMaximo - 1; j >= 0; j--) {
		           convertedDistances = new double[origDistances.length];
		           System.arraycopy(origDistances, 0, convertedDistances, 0, origDistances.length);
			 	   double [] distribution = makeDistribution(neighbours,  convertedDistances);
		           double thisPrediction = Utils.maxIndex(distribution);
			 	   if (Train.classAttribute().isNumeric()) {
			 	    thisPrediction = distribution[0];
			 	    double err = thisPrediction - instance.classValue();
			 	    performanceStatsSq[j] += err * err;   // Squared error
			 	    performanceStats[j] += Math.abs(err); // Absolute error
			 	   } else {
			 	    if (thisPrediction != instance.classValue()) {
			 	      performanceStats[j] ++;             // Classification error
			 	    }
			 	   }
			 	   if (j >= 1) {
			 	    neighbours = pruneToK(neighbours, convertedDistances, j);
			 	   }
			 	}
		    }
			 
			for(int i = 0; i < kNNMaximo; i++) {
			 	if (m_Debug) {
			 	  System.err.print("Hold-one-out performance of " + (i + 1) + " neighbors " );
			 	}
			 	if (Train.classAttribute().isNumeric()) {
			 	  if (m_Debug) {
			 	    if (MeanSquared) {
			 	      System.err.println("(RMSE) = " + Math.sqrt(performanceStatsSq[i] / Train.numInstances()));
				    } else {
			 	      System.err.println("(MAE) = " + performanceStats[i] / Train.numInstances());
			 	    }
			 	  }
			 	}else{
			 	  if (m_Debug) {
			 	    System.err.println("(%ERR) = " + 100.0 * performanceStats[i] / Train.numInstances());
			 	  }	
			 	}
		    }
			       
		    double [] searchStats = performanceStats;
		    if (Train.classAttribute().isNumeric() && MeanSquared) {
			   searchStats = performanceStatsSq;
		    }
			double bestPerformance = Double.NaN;
			int bestK = 1;
		   	for(int i = 0; i < kNNMaximo; i++) {
				if (Double.isNaN(bestPerformance) || (bestPerformance > searchStats[i])) {
				  bestPerformance = searchStats[i];
				  bestK = i + 1;
				}
		   	}
		   	kNN = bestK;
			if (m_Debug) {
				  System.err.println("Selected k = " + bestK);
			}
			kNNvalido = true;
			}catch (Exception ex) {
			  throw new Error("Couldn't optimize by cross-validation: "
				      +ex.getMessage());
			}
   	}
    public String globalInfo() {
		
	     return  "K-nearest neighbours classifier. Can "
	       + "select appropriate value of K based on cross-validation. Can also do "
	       + "distance weighting.\n\n"
	       + "For more information, see\n\n"
	       + getTechnicalInformation().toString();
	}
	  
	public TechnicalInformation getTechnicalInformation() {
	     
		 TechnicalInformation 	result;
	     result = new TechnicalInformation(Type.ARTICLE);
	     result.setValue(Field.AUTHOR, "D. Aha and D. Kibler");
	     result.setValue(Field.YEAR, "1991");
	     result.setValue(Field.TITLE, "Instance-based learning algorithms");
	     result.setValue(Field.JOURNAL, "Machine Learning");
	     result.setValue(Field.VOLUME, "6");
	     result.setValue(Field.PAGES, "37-66");
	     
	     return result;
	}
	
	  
	  	  

   	
    public String 	KNNTipText() {
         return "The number of neighbours to use.";
    }
    
   	public String getRevision() {
   	    return RevisionUtils.extract("$Revision: 6572 $");
   	}
   	
	public static void main(String [] argv) {
		runClassifier(new NuestroModelo(), argv);
	}
}	
