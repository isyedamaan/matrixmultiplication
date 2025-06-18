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

                // ForkJoinPool
                MatrixMultiplier forkJoin = new ForkJoinMatrixMultiplier(threadCount);
                long startFJ = System.nanoTime();
                double[][] resultFJ = forkJoin.multiply(matrixA, matrixB);
                long endFJ = System.nanoTime();
                double timeFJ = (endFJ - startFJ) / 1e6;
                System.out.printf("%s Time: %.2f ms%n", forkJoin.getName(), timeFJ);

                boolean correctFJ = MatrixUtils.areMatricesEqual(resultSeq, resultFJ, tolerance);
                System.out.printf("Results Match (%s): %s%n", forkJoin.getName(), correctFJ ? "YES" : "NO");

                // ParallelStream
                MatrixMultiplier parallelStream = new ParallelStreamMatrixMultiplier();
                long startPS = System.nanoTime();
                double[][] resultPS = parallelStream.multiply(matrixA, matrixB);
                long endPS = System.nanoTime();
                double timePS = (endPS - startPS) / 1e6;
                System.out.printf("%s Time: %.2f ms%n", parallelStream.getName(), timePS);

                boolean correctPS = MatrixUtils.areMatricesEqual(resultSeq, resultPS, tolerance);
                System.out.printf("Results Match (%s): %s%n", parallelStream.getName(), correctPS ? "YES" : "NO");

                // BlockedParallel (ForkJoin)
                MatrixMultiplier blockedParallel = new BlockedParallelMatrixMultiplier(threadCount, 64);
                long startBP = System.nanoTime();
                double[][] resultBP = blockedParallel.multiply(matrixA, matrixB);
                long endBP = System.nanoTime();
                double timeBP = (endBP - startBP) / 1e6;
                System.out.printf("%s Time: %.2f ms%n", blockedParallel.getName(), timeBP);

                boolean correctBP = MatrixUtils.areMatricesEqual(resultSeq, resultBP, tolerance);
                System.out.printf("Results Match (%s): %s%n", blockedParallel.getName(), correctBP ? "YES" : "NO");

                System.out.println();
            }

            System.out.println();
        }
    }
}
