package matrixmultiplication;

public class MatrixMultiplicationTest {

    public static void main(String[] args) {
        int[] sizes = {500, 1000, 1500};
        int[] threadCounts = {4, 8, 16, 32}; // Thread pool sizes to compare
        double tolerance = 1e-6;

        for (int size : sizes) {
            System.out.println("===== Matrix Size: " + size + " x " + size + " =====");

            // Generate random matrices
            double[][] matrixA = MatrixUtils.generateRandomMatrix(size, size);
            double[][] matrixB = MatrixUtils.generateRandomMatrix(size, size);

            // Sequential multiplication
            MatrixMultiplier sequential = new SequentialMatrixMultiplier();
            long startSeq = System.nanoTime();
            double[][] resultSeq = sequential.multiply(matrixA, matrixB);
            long endSeq = System.nanoTime();
            double timeSeq = (endSeq - startSeq) / 1e6;
            System.out.printf("Sequential Time: %.2f ms%n", timeSeq);

            // Concurrent multiplication with different thread counts
            for (int threadCount : threadCounts) {
                // Manual Threads
                MatrixMultiplier manual = new ConcurrentMatrixMultiplier(threadCount);
                long startManual = System.nanoTime();
                double[][] resultManual = manual.multiply(matrixA, matrixB);
                long endManual = System.nanoTime();
                double timeManual = (endManual - startManual) / 1e6;
                System.out.printf("%s Time: %.2f ms%n", manual.getName(), timeManual);

                boolean correctManual = MatrixUtils.areMatricesEqual(resultSeq, resultManual, tolerance);
                System.out.printf("Results Match (%s): %s%n", manual.getName(), correctManual ? "YES" : "NO");

                // Thread Pool
                MatrixMultiplier threadPool = new ThreadPoolMatrixMultiplier(threadCount);
                long startTP = System.nanoTime();
                double[][] resultTP = threadPool.multiply(matrixA, matrixB);
                long endTP = System.nanoTime();
                double timeTP = (endTP - startTP) / 1e6;
                System.out.printf("%s Time: %.2f ms%n", threadPool.getName(), timeTP);

                boolean correctTP = MatrixUtils.areMatricesEqual(resultSeq, resultTP, tolerance);
                System.out.printf("Results Match (%s): %s%n", threadPool.getName(), correctTP ? "YES" : "NO");

                System.out.println();
            }

            System.out.println();
        }
    }
}
