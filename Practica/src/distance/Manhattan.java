package distance;

public class Manhattan implements Distance {
	private double[] a;
	private int pos = 0;
	
	public Manhattan(int numberOfAttr) {
		this.a = new double[numberOfAttr];
	}
	public double getDistance() {
		double sum = 0;
		for (double el: this.a) sum += el;
		return sum;
	}

	public void setAtributeDist(double a, double b) {
		this.a[this.pos++] = Math.abs(a-b);
	}
}
