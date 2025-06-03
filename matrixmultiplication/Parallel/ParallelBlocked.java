package matrixmultiplication.Parallel;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.*;
import java.util.concurrent.ConcurrentSkipListSet;

public class ParallelBlocked {

    public static int[][] generateRandomMatrix(int n) {
        Random rand = new Random();
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i][j] = rand.nextInt(10);
        return matrix;
    }

    public static int[][] multiplyNaive(int[][] A, int[][] B, int n) {
        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    public static boolean validateResult(int[][] C1, int[][] C2, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (C1[i][j] != C2[i][j]) {
                    System.out.printf("Mismatch at [%d][%d]: %d != %d%n", i, j, C1[i][j], C2[i][j]);
                    return false;
                }
            }
        }
        return true;
    }

    private static void multiplyBlockedSection(int[][] A, int[][] B, int[][] C, int n,
            int BLOCK_SIZE, int iStart, int iEnd) {
        for (int iBlock = iStart; iBlock < iEnd; iBlock += BLOCK_SIZE) {
            int iMax = Math.min(iBlock + BLOCK_SIZE, iEnd);
            for (int jBlock = 0; jBlock < n; jBlock += BLOCK_SIZE) {
                int jMax = Math.min(jBlock + BLOCK_SIZE, n);
                for (int kBlock = 0; kBlock < n; kBlock += BLOCK_SIZE) {
                    int kMax = Math.min(kBlock + BLOCK_SIZE, n);

                    for (int i = iBlock; i < iMax; i++) {
                        for (int j = jBlock; j < jMax; j++) {
                            int sum = 0;
                            for (int k = kBlock; k < kMax; k++) {
                                sum += A[i][k] * B[k][j];
                            }
                            C[i][j] += sum;
                        }
                    }
                }
            }
        }
    }

    // Parallel blocked multiplication with thread CPU time tracking
    public static long multiplyBlockedParallel(int[][] A, int[][] B, int[][] C, int n, int BLOCK_SIZE,
            ThreadMXBean threadMXBean) throws InterruptedException {
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        Set<Long> threadIds = new ConcurrentSkipListSet<>();

        List<Future<?>> futures = new ArrayList<>();
        int rowsPerThread = (n + numThreads - 1) / numThreads; // ceiling division

        for (int t = 0; t < numThreads; t++) {
            final int iStart = t * rowsPerThread;
            final int iEnd = Math.min(iStart + rowsPerThread, n);

            if (iStart >= iEnd)
                break;

            futures.add(executor.submit(() -> {
                // Capture thread ID
                threadIds.add(Thread.currentThread().getId());
                multiplyBlockedSection(A, B, C, n, BLOCK_SIZE, iStart, iEnd);
            }));
        }

        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }

        executor.shutdown();

        // Sum CPU time for all worker threads
        long totalCpuTimeNs = 0;
        for (long tid : threadIds) {
            if (threadMXBean.isThreadCpuTimeSupported() && threadMXBean.isThreadCpuTimeEnabled()) {
                long time = threadMXBean.getThreadCpuTime(tid);
                if (time != -1) { // -1 means thread not alive or unsupported
                    totalCpuTimeNs += time;
                }
            }
        }

        return totalCpuTimeNs;
    }

    public static String getMemoryUsageMB() {
        Runtime runtime = Runtime.getRuntime();
        long used = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
        long free = runtime.freeMemory() / (1024 * 1024);
        long total = runtime.totalMemory() / (1024 * 1024);
        return String.format("Memory Usage: Used = %d MB, Free = %d MB, Total = %d MB", used, free, total);
    }

    public static void main(String[] args) throws InterruptedException {
        int[] sizes = { 500, 1000, 2000, 4000 };
        int BLOCK_SIZE = 128;

        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        boolean cpuTimeSupported = threadMXBean.isThreadCpuTimeSupported();
        if (cpuTimeSupported)
            threadMXBean.setThreadCpuTimeEnabled(true);

        System.out.println("Running matrix multiplication with parallel implementation by block");
        System.out.println("Block size: " + BLOCK_SIZE);
        System.out.println("---------------------------------------------------\n");

        for (int n : sizes) {
            System.out.println("Matrix size: " + n + "x" + n);

            int[][] A = generateRandomMatrix(n);
            int[][] B = generateRandomMatrix(n);
            int[][] C = new int[n][n];

            System.gc();

            long beforeUsedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            long startTime = System.nanoTime();

            long totalCpuTimeNs = multiplyBlockedParallel(A, B, C, n, BLOCK_SIZE, threadMXBean);

            long endTime = System.nanoTime();
            long afterUsedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

            long executionTimeMs = (endTime - startTime) / 1_000_000;
            long memoryUsedMB = (afterUsedMem - beforeUsedMem) / (1024 * 1024);
            long cpuTimeMs = totalCpuTimeNs / 1_000_000;

            System.out.println("Execution time: " + executionTimeMs + " ms");
            System.out.println("Memory used during operation: " + memoryUsedMB + " MB");

            if (cpuTimeSupported) {
                System.out.println("CPU time used by worker threads: " + cpuTimeMs + " ms");
            } else {
                System.out.println("CPU time measurement not supported.");
            }

            System.out.println(getMemoryUsageMB());

            /*
             * System.out.print("Validating result... ");
             * int[][] reference = multiplyNaive(A, B, n);
             * boolean valid = validateResult(C, reference, n);
             * System.out.println(valid ? "PASSED" : "FAILED");
             */

            System.out.println("---------------------------------------------------\n");
        }
    }
}
