package matrixmultiplication.Parallel;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Random;
import java.util.stream.IntStream;

//uses parallelStream API, IntStream.range().parallel()
public class ParallelStreamMatrixMultiplier {

    public static void main(String[] args) {
        int[] sizes = { 500, 1000, 2000 };

        for (int size : sizes) {
            double[][] A = generateRandomMatrix(size);
            double[][] B = generateRandomMatrix(size);
            multiplyAndBenchmark(A, B);
        }
    }

    public static void multiplyAndBenchmark(double[][] matrixA, double[][] matrixB) {
        int rowsA = matrixA.length;
        int colsA = matrixA[0].length;
        int rowsB = matrixB.length;
        int colsB = matrixB[0].length;

        if (colsA != rowsB) {
            throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication.");
        }

        double[][] result = new double[rowsA][colsB];

        Runtime runtime = Runtime.getRuntime();
        System.out.println("Running matrix multiplication with parallelStream implementation");
        System.out.println("---------------------------------------------------");

        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
        }

        long beforeUsedMem = runtime.totalMemory() - runtime.freeMemory();
        long beforeCpuTime = getCpuTimeOfAllThreads();
        long startTime = System.currentTimeMillis();

        IntStream.range(0, rowsA).parallel().forEach(i -> {
            for (int j = 0; j < colsB; j++) {
                double sum = 0;
                for (int k = 0; k < colsA; k++) {
                    sum += matrixA[i][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        });

        long endTime = System.currentTimeMillis();
        long afterUsedMem = runtime.totalMemory() - runtime.freeMemory();
        long afterCpuTime = getCpuTimeOfAllThreads();

        System.out.println("Matrix size: " + rowsA + "x" + colsB);
        System.out.println("Execution time: " + (endTime - startTime) + " ms");
        System.out.printf("Memory used during operation: %.0f MB%n",
                (afterUsedMem - beforeUsedMem) / (1024.0 * 1024.0));

        if (beforeCpuTime != -1 && afterCpuTime != -1) {
            System.out.printf("CPU time used by worker threads: %.0f ms%n",
                    (afterCpuTime - beforeCpuTime) / 1_000_000.0);
        } else {
            System.out.println("CPU time measurement not supported.");
        }

        printMemoryUsage();
        System.out.println("---------------------------------------------------\n");
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
