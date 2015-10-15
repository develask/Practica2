package distance;

import java.util.ArrayList;

public class Minkowski{

	private ArrayList<Double> a;
	private int p = 3;
	
	public Minkowski(int p) {
		this.p = p>0?p:3;
	}
	
	public double getDistance() {
		double sum = 0;
		for (double el: this.a) sum += el;
		return Math.pow(sum, 1.0/this.p);
	}

	public void setAtributeDist(double a, double b) {
		this.a.add(Math.pow(Math.abs(a-b), this.p));
	}

}
