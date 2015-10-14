package distance;

public class Minkowski implements Distance{

	private double[] a;
	private int pos = 0;
	private int p = 3; // no se que es esto
	
	public Minkowski(int numberOfAttr) {
		this.a = new double[numberOfAttr];
	}
	
	@Override
	public double getDistance() {
		double sum = 0;
		for (double el: this.a) sum += el;
		return Math.pow(sum, 1.0/this.p);
	}

	@Override
	public void setAtributeDist(double a, double b) {
		this.a[this.pos++] = Math.pow(Math.abs(a-b), this.p);
	}

}
