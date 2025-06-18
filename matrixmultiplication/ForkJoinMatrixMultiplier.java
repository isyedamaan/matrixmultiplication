package matrixmultiplication;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ForkJoinMatrixMultiplier implements MatrixMultiplier {
    private int parallelism;

    public ForkJoinMatrixMultiplier(int parallelism) {
        this.parallelism = parallelism;
    }

    public ForkJoinMatrixMultiplier() {
        this(Runtime.getRuntime().availableProcessors());
    }

    @Override
    public double[][] multiply(double[][] matrixA, double[][] matrixB) {
        int rowsA = matrixA.length;
        int colsA = matrixA[0].length;
        int rowsB = matrixB.length;
        int colsB = matrixB[0].length;

        if (colsA != rowsB) {
            throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication");
        }

        double[][] result = new double[rowsA][colsB];
        ForkJoinPool pool = new ForkJoinPool(parallelism);
        pool.invoke(new MultiplyTask(matrixA, matrixB, result, 0, rowsA));
        pool.shutdown();
        return result;
    }

    private static class MultiplyTask extends RecursiveAction {
        private static final int THRESHOLD = 64; // Tune for best performance
        private final double[][] matrixA, matrixB, result;
        private final int startRow, endRow;

        MultiplyTask(double[][] matrixA, double[][] matrixB, double[][] result, int startRow, int endRow) {
            this.matrixA = matrixA;
            this.matrixB = matrixB;
            this.result = result;
            this.startRow = startRow;
            this.endRow = endRow;
        }

        @Override
        protected void compute() {
            if (endRow - startRow <= THRESHOLD) {
                int colsA = matrixA[0].length;
                int colsB = matrixB[0].length;
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < colsB; j++) {
                        double sum = 0;
                        for (int k = 0; k < colsA; k++) {
                            sum += matrixA[i][k] * matrixB[k][j];
                        }
                        result[i][j] = sum;
                    }
                }
            } else {
                int mid = (startRow + endRow) / 2;
                MultiplyTask left = new MultiplyTask(matrixA, matrixB, result, startRow, mid);
                MultiplyTask right = new MultiplyTask(matrixA, matrixB, result, mid, endRow);
                invokeAll(left, right);
            }
        }
    }

    @Override
    public String getName() {
        return "ForkJoinPool (" + parallelism + " threads)";
    }
} 