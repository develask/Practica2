package src;

import java.util.Hashtable;
import src.NuestroModelo.DistanceWight;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Probador {
	
	public static void main(String[] args) {
		
		Hashtable<String, String> params =Args.parse(args);
		int k=1;
		DistanceWight w = DistanceWight.NoDistance;
		int wWeka = IBk.WEIGHT_NONE;
		boolean selectedWeka = false;
		boolean best=true;
		int t = 3;
		int porcent = 70;
		Instances instancias = null, instanciasparaclasificar = null;
		try {
			String tmp = params.get("-best");
			best = tmp != null;
			tmp = params.get("-model");
			selectedWeka = (tmp!= null && tmp.equals("weka"));
			tmp = params.get("-k");
			if (tmp!=null && Integer.parseInt(tmp)>0){
				k = Integer.parseInt(tmp);
			}
			tmp = params.get("-w");
			switch (tmp!=null?tmp:"1") {
			case "1":
			case "NoDistance":
				w = DistanceWight.NoDistance;
				wWeka = IBk.WEIGHT_NONE;
				break;
			case "2":
			case "OneDivDistance":
				w = DistanceWight.OneDivDistance;
				wWeka = IBk.WEIGHT_INVERSE;
				break;
			}
			tmp = params.get("-m");
			switch (tmp!=null?tmp:"3") {
			case "Manhattan":
				t = 1;
				break;
			case "Euclidea":
				t = 2;
				break;
			default:
				if (tmp!=null && Integer.parseInt(tmp)>0){
					t = Integer.parseInt(tmp);
				}
			}
			Instances ins = Lector.getLector().leerInstancias(params.get("-f"));
			if (ins == null){
				System.exit(1);
			}
			tmp = params.get("-holdout");
			if (tmp!=null && Integer.parseInt(tmp)>0){
				porcent = Integer.parseInt(tmp);
			}
			int trainSize = (int) Math.round(ins.numInstances() * porcent/100);
			instancias = new Instances(ins, 0, trainSize);
			instanciasparaclasificar = new Instances(ins, trainSize,ins.numInstances()-trainSize);
		}catch(Exception e){
			System.out.println("Ha habido algun problema con los parametros.\n");
			e.printStackTrace();
			System.exit(1);
		}
		Modelo nm;
		if (best){
			String kmax = params.get("-kmax");
			int kmaxvalue = instancias.numInstances();
			if (kmax != null && Integer.parseInt(kmax)>0 && Integer.parseInt(kmax)<kmaxvalue){
				kmaxvalue = Integer.parseInt(kmax);
			}
			double mejor = 0;
			String r = "";
			System.out.println("Se va a proceder a selecionar el mejor evaluador:");
			System.out.println("kmax: "+kmaxvalue);
			int porcentageBar = 0;
			ProgressBar.updateProgress(porcentageBar);
			long tStart = System.currentTimeMillis();
			 for (int k2=1;k2<kmaxvalue; k2+=1){
				 int nuevop = k2*100/kmaxvalue;
				 if (nuevop>porcentageBar){
					 porcentageBar = nuevop;
					 ProgressBar.updateProgress(porcentageBar/100.0);
				 }
				 for (int dw = 1; dw <= 2; dw++) {
					 if (dw==1){
						 w = DistanceWight.NoDistance;
						wWeka = IBk.WEIGHT_NONE;
					 }else{ //dw=2
						 w = DistanceWight.OneDivDistance;
						wWeka = IBk.WEIGHT_INVERSE;
					 }
					 for (int dt = 1; dt <= 3; dt++) {
						 if (selectedWeka){
							 nm = new ModeloWeka(k2, wWeka, dt);
						}else{
							nm = new NuestroModelo(k2,w,dt);
						}
						nm.buildClasifier(instancias);
						nm.evaluarModelo(instanciasparaclasificar);
						double nueva = nm.accuracy();
					    if (nueva > mejor){
					    	mejor = nueva;
					    	r = nm.calcularMediciones();
					    }
					}
				 }
			 }
			 long tEnd = System.currentTimeMillis();
			 long tTrans = tEnd - tStart;
			 double elapsedSeconds = tTrans / 1000.0;
			 System.out.println("\nTiempo: " +elapsedSeconds+ " segundos\n");
			 System.out.println(r);
		}else{
			System.out.println("Se esta ejecutando el modelo seleccionado.");
			long tStart = System.currentTimeMillis();
			if (selectedWeka){
				nm = new ModeloWeka(k, wWeka, t);
			}else{
				nm = new NuestroModelo(k,w,t);
			}
			nm.buildClasifier(instancias);
			nm.evaluarModelo(instanciasparaclasificar);
			long tEnd = System.currentTimeMillis();
			long tTrans = tEnd - tStart;
			double elapsedSeconds = tTrans / 1000.0;
			System.out.println("Tiempo: " +elapsedSeconds+ " segundos\n");
			System.out.println(nm.calcularMediciones());
		}

	}

}
