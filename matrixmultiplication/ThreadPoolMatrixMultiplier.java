package matrixmultiplication;


import java.util.concurrent.*;

public class ThreadPoolMatrixMultiplier implements MatrixMultiplier {

    private int numThreads;

    public ThreadPoolMatrixMultiplier(int numThreads) {
        this.numThreads = numThreads;
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

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch latch = new CountDownLatch(rowsA);

        for (int i = 0; i < rowsA; i++) {
            final int row = i;
            executor.submit(() -> {
                for (int j = 0; j < colsB; j++) {
                    double sum = 0;
                    for (int k = 0; k < colsA; k++) {
                        sum += matrixA[row][k] * matrixB[k][j];
                    }
                    result[row][j] = sum;
                }
                latch.countDown();
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("ThreadPool matrix multiplication interrupted", e);
        } finally {
            executor.shutdown();
        }

        return result;
    }

    @Override
    public String getName() {
        return "ThreadPool (" + numThreads + " threads)";
    }
}
