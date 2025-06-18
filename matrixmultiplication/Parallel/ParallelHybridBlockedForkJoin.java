package matrixmultiplication.Parallel;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

class ForkJoinTaskRecursive extends RecursiveAction {
    // change THRESHOLD = 128 → 64 / 256 to compare performance trade-offs
    static final int THRESHOLD = 128;

    double[][] A, B, C;
    int rowA, colA, rowB, colB, rowC, colC, size;

    public ForkJoinTaskRecursive(double[][] A, double[][] B, double[][] C,
            int rowA, int colA,
            int rowB, int colB,
            int rowC, int colC,
            int size) {
        this.A = A;
        this.B = B;
        this.C = C;
        this.rowA = rowA;
        this.colA = colA;
        this.rowB = rowB;
        this.colB = colB;
        this.rowC = rowC;
        this.colC = colC;
        this.size = size;
    }

    @Override
    protected void compute() {
        if (size <= THRESHOLD) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    double sum = 0;
                    for (int k = 0; k < size; k++) {
                        sum += A[rowA + i][colA + k] * B[rowB + k][colB + j];
                    }
                    C[rowC + i][colC + j] += sum;
                }
            }
        } else {
            int half = size / 2;
            invokeAll(
                    new ForkJoinTaskRecursive(A, B, C, rowA, colA, rowB, colB, rowC, colC, half),
                    new ForkJoinTaskRecursive(A, B, C, rowA, colA + half, rowB + half, colB, rowC, colC, half),
                    new ForkJoinTaskRecursive(A, B, C, rowA, colA, rowB, colB + half, rowC, colC + half, half),
                    new ForkJoinTaskRecursive(A, B, C, rowA, colA + half, rowB + half, colB + half, rowC, colC + half,
                            half),
                    new ForkJoinTaskRecursive(A, B, C, rowA + half, colA, rowB, colB, rowC + half, colC, half),
                    new ForkJoinTaskRecursive(A, B, C, rowA + half, colA + half, rowB + half, colB, rowC + half, colC,
                            half),
                    new ForkJoinTaskRecursive(A, B, C, rowA + half, colA, rowB, colB + half, rowC + half, colC + half,
                            half),
                    new ForkJoinTaskRecursive(A, B, C, rowA + half, colA + half, rowB + half, colB + half, rowC + half,
                            colC + half, half));
        }
    }
}

public class ParallelHybridBlockedForkJoin {

    public static void main(String[] args) {
        int[] sizes = { 500, 1000, 2000 };

        for (int size : sizes) {
            double[][] A = generateRandomMatrix(size);
            double[][] B = generateRandomMatrix(size);
            double[][] result = new double[size][size];

            System.out.println("Running matrix multiplication with ForkJoin (block-based)");
            System.out.println("---------------------------------------------------");

            Runtime runtime = Runtime.getRuntime();
            System.gc();
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
            }

            long beforeUsedMem = runtime.totalMemory() - runtime.freeMemory();
            long beforeCpuTime = getCpuTimeOfAllThreads();
            long startTime = System.currentTimeMillis();

            ForkJoinPool pool = new ForkJoinPool();
            pool.invoke(new ForkJoinTaskRecursive(A, B, result, 0, 0, 0, 0, 0, 0, size));
            pool.shutdown();

            long endTime = System.currentTimeMillis();
            long afterUsedMem = runtime.totalMemory() - runtime.freeMemory();
            long afterCpuTime = getCpuTimeOfAllThreads();

            System.out.println("Matrix size: " + size + "x" + size);
            System.out.println("Execution time: " + (endTime - startTime) + " ms");
            System.out.printf("Memory used during operation: %.0f MB%n",
                    (afterUsedMem - beforeUsedMem) / (1024.0 * 1024.0));

            if (beforeCpuTime != -1 && afterCpuTime != -1) {
                long cpuTimeNano = afterCpuTime - beforeCpuTime;
                System.out.printf("CPU time used by worker threads: %.0f ms%n", cpuTimeNano / 1_000_000.0);
            } else {
                System.out.println("CPU time measurement not supported.");
            }

            printMemoryUsage();
            System.out.println("---------------------------------------------------\n");
        }
    }

    public static double[][] generateRandomMatrix(int size) {
        double[][] matrix = new double[size][size];
        Random rand = new Random();
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                matrix[i][j] = rand.nextDouble() * 10;
        return matrix;
    }

    private static long getCpuTimeOfAllThreads() {
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        if (!threadMXBean.isThreadCpuTimeSupported() || !threadMXBean.isThreadCpuTimeEnabled()) {
            return -1;
        }
        long totalCpuTime = 0;
        for (long id : threadMXBean.getAllThreadIds()) {
            long time = threadMXBean.getThreadCpuTime(id);
            if (time != -1) {
                totalCpuTime += time;
            }
        }
        return totalCpuTime;
    }

    private static void printMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long total = runtime.totalMemory();
        long free = runtime.freeMemory();
        long used = total - free;
        System.out.printf("Memory Usage: Used = %.0f MB, Free = %.0f MB, Total = %.0f MB%n",
                used / (1024.0 * 1024.0),
                free / (1024.0 * 1024.0),
                total / (1024.0 * 1024.0));
    }
}
