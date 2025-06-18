package matrixmultiplication;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ForkJoin {

    static final int BLOCK_SIZE = 128;
    static ForkJoinPool fjPool = new ForkJoinPool();

    static class MultiplyTask extends RecursiveAction {
        private final double[][] A, B, C;
        private final int rowStart, rowEnd, colStart, colEnd;

        MultiplyTask(double[][] A, double[][] B, double[][] C,
                     int rowStart, int rowEnd, int colStart, int colEnd) {
            this.A = A; this.B = B; this.C = C;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.colStart = colStart; this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            int rowCount = rowEnd - rowStart;
            int colCount = colEnd - colStart;

            if (rowCount <= BLOCK_SIZE && colCount <= BLOCK_SIZE) {
                int n = A.length;
                for (int i = rowStart; i < rowEnd; i++) {
                    for (int j = colStart; j < colEnd; j++) {
                        double sum = 0;
                        for (int k = 0; k < n; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            } else if (rowCount >= colCount) {
                int mid = (rowStart + rowEnd) / 2;
                invokeAll(
                    new MultiplyTask(A, B, C, rowStart, mid, colStart, colEnd),
                    new MultiplyTask(A, B, C, mid, rowEnd, colStart, colEnd)
                );
            } else {
                int mid = (colStart + colEnd) / 2;
                invokeAll(
                    new MultiplyTask(A, B, C, rowStart, rowEnd, colStart, mid),
                    new MultiplyTask(A, B, C, rowStart, rowEnd, mid, colEnd)
                );
            }
        }
    }

    static double[][] generateMatrix(int size) {
        double[][] matrix = new double[size][size];
        Random rnd = new Random();
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                matrix[i][j] = rnd.nextDouble();
        return matrix;
    }

    public static void main(String[] args) {
        int[] sizes = {500, 1000, 2000, 4000};

        System.out.println("Running matrix multiplication with parallel implementation using ForkJoin");
        System.out.println("Block size: "+BLOCK_SIZE);
        System.out.println("---------------------------------------------------\n");

        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        boolean isCpuTimeSupported = threadMXBean.isThreadCpuTimeSupported();
        if (isCpuTimeSupported && !threadMXBean.isThreadCpuTimeEnabled()) {
            threadMXBean.setThreadCpuTimeEnabled(true);
        }

        for (int size : sizes) {
            System.out.println("Matrix size: " + size + "x" + size);

            System.gc();
            long beforeUsedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

            double[][] A = generateMatrix(size);
            double[][] B = generateMatrix(size);
            double[][] C = new double[size][size];

            long startWallTime = System.currentTimeMillis();
            long startCpuTime = isCpuTimeSupported ? threadMXBean.getCurrentThreadCpuTime() : 0;

            fjPool.invoke(new MultiplyTask(A, B, C, 0, size, 0, size));

            long endCpuTime = isCpuTimeSupported ? threadMXBean.getCurrentThreadCpuTime() : 0;
            long endWallTime = System.currentTimeMillis();

            long afterUsedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

            long execTimeMs = endWallTime - startWallTime;
            long cpuTimeNs = endCpuTime - startCpuTime;
            long cpuTimeMs = cpuTimeNs / 1_000_000;

            long memUsedDuringOp = (afterUsedMem - beforeUsedMem) / (1024 * 1024);

            System.out.println("Execution time (wall clock): " + execTimeMs + " ms");
            System.out.println("Memory used during operation: " + memUsedDuringOp + " MB");
            System.out.println("CPU time used by current thread: " + cpuTimeMs + " ms");

            long totalMemMB = Runtime.getRuntime().totalMemory() / (1024 * 1024);
            long freeMemMB = Runtime.getRuntime().freeMemory() / (1024 * 1024);
            long usedMemMB = totalMemMB - freeMemMB;
            System.out.printf("Memory Usage: Used = %d MB, Free = %d MB, Total = %d MB%n", usedMemMB, freeMemMB, totalMemMB);

            System.out.println("---------------------------------------------------");
        }
    }
}
