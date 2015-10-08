package src;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class PreProcesador {

	public static void main(String[] args) {
		
		Instances instanciasParaFiltrar [] = new Instances [2];
		instanciasParaFiltrar[0] = Lector.getLector().leerInstancias(args[0]);
		instanciasParaFiltrar[1] = Lector.getLector().leerInstancias(args[0]);
		
		// Crear filtro
		Integer cortar;
		try {
		 cortar= Integer.parseInt(args[3]);
		 cortar = (cortar>100) ? 100:cortar;
		 cortar = (cortar<0) ? 0:cortar;
		} catch (Exception e) {
			cortar=100;
		}
		
		int trainSize = (int) Math.round(instanciasParaFiltrar[0].numInstances() * cortar/100);
	    instanciasParaFiltrar[0] = new Instances(instanciasParaFiltrar[0], 0, trainSize);
	    instanciasParaFiltrar[1] = new Instances(instanciasParaFiltrar[1], trainSize,(int) Math.round(instanciasParaFiltrar[0].numInstances()));
	    System.out.println("Train: "+instanciasParaFiltrar[0].numInstances()+" instancias");
	    System.out.println("Development: "+instanciasParaFiltrar[1].numInstances()+" instancias");
	    
	    
		InterquartileRange interquartile = new InterquartileRange();
		interquartile.setExtremeValuesFactor(6.0);
		interquartile.setOutlierFactor(3.0);
		try {
			interquartile.setInputFormat(instanciasParaFiltrar[0]);
			instanciasParaFiltrar[0] = Filter.useFilter(instanciasParaFiltrar[0], interquartile);
			instanciasParaFiltrar[1] = Filter.useFilter(instanciasParaFiltrar[1], interquartile);
			
			RemoveWithValues removeWithValues = new RemoveWithValues();
			removeWithValues.setAttributeIndex("last");
			removeWithValues.setNominalIndices("last");
			removeWithValues.setInputFormat(instanciasParaFiltrar[0]);
			instanciasParaFiltrar[0] = Filter.useFilter(instanciasParaFiltrar[0], removeWithValues);
			instanciasParaFiltrar[1] = Filter.useFilter(instanciasParaFiltrar[1], removeWithValues);
			removeWithValues = new RemoveWithValues();
			removeWithValues.setAttributeIndex(""+(instanciasParaFiltrar[0].numAttributes()-1));
			removeWithValues.setNominalIndices("last");
			removeWithValues.setInputFormat(instanciasParaFiltrar[0]);
			instanciasParaFiltrar[0] = Filter.useFilter(instanciasParaFiltrar[0], removeWithValues);
			instanciasParaFiltrar[1] = Filter.useFilter(instanciasParaFiltrar[1], removeWithValues);
			
			Remove remove = new Remove();
			remove.setInputFormat(instanciasParaFiltrar[0]);
			remove.setAttributeIndicesArray(new int[] {instanciasParaFiltrar[0].numAttributes()-1,instanciasParaFiltrar[0].numAttributes()});
			instanciasParaFiltrar[0] = Filter.useFilter(instanciasParaFiltrar[0], remove);
			instanciasParaFiltrar[1] = Filter.useFilter(instanciasParaFiltrar[1], remove);
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		
		
	    
	    //multiFilter
		MultiFilter multiFilter = new MultiFilter();
		Filter[] filtros = new Filter[3];
		Discretize discretize = new Discretize();
		Randomize randomize = new Randomize();
		RemoveUseless removeUseless = new RemoveUseless();
		
		filtros[0] = discretize;
		filtros[1] = randomize;
		filtros[2] = removeUseless;


		
		multiFilter.setFilters(filtros);	
		Instances instanciasParaFiltrar2 = instanciasParaFiltrar[0];
		try {
			multiFilter.setInputFormat(instanciasParaFiltrar2);
			instanciasParaFiltrar2 = Filter.useFilter(instanciasParaFiltrar2, multiFilter);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		AttributeSelection filter= new AttributeSelection();
		// Crear tipo de filtro
		CfsSubsetEval eval = new CfsSubsetEval();
		// Crear metodo de busqueda
		GreedyStepwise search=new GreedyStepwise();
		// añadir lo creado al filtro
		filter.setEvaluator(eval);
		filter.setSearch(search);
				
		Instances instanciasParaFiltrar3 [] =new Instances [2];
				
		try {
			filter.setInputFormat(instanciasParaFiltrar[0]);
			instanciasParaFiltrar3[0] = Filter.useFilter(instanciasParaFiltrar2, filter);
			instanciasParaFiltrar3[1] = Filter.useFilter(instanciasParaFiltrar[1], filter);
		} catch (Exception e) {
			e.printStackTrace();
		}
		Escritor.getEscritor().escribirInstancias(instanciasParaFiltrar3[0], args[2]);
		Escritor.getEscritor().escribirInstancias(instanciasParaFiltrar3[1], args[3]);
	}

}
