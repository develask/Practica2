package distance;

public class Euclidea implements Distance{
	private double[] a;
	private int pos = 0;
	
	public Euclidea(int numberOfAttr) {
		this.a = new double[numberOfAttr];
	}
	public double getDistance() {
		double sum = 0;
		for (double el: this.a) sum += el;
		return Math.sqrt(sum);
	}

	public void setAtributeDist(double a, double b) {
		this.a[this.pos++] = Math.pow(a-b, 2);
	}

}
