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
                MatrixMultiplier concurrent = new ConcurrentMatrixMultiplier(threadCount);
                long startCon = System.nanoTime();
                double[][] resultCon = concurrent.multiply(matrixA, matrixB);
                long endCon = System.nanoTime();
                double timeCon = (endCon - startCon) / 1e6;
                System.out.printf("Concurrent Time (%d threads): %.2f ms%n", threadCount, timeCon);

                // Verify correctness
                boolean correct = MatrixUtils.areMatricesEqual(resultSeq, resultCon, tolerance);
                System.out.println("Results Match (" + threadCount + " threads): " + (correct ? "YES" : "NO"));
            }

            System.out.println();
        }
    }
}
