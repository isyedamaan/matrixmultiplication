package matrixmultiplication;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Random;
import java.util.concurrent.*;

public class ParallelImplementation {

    public static int[][] generateRandomMatrix(int n) {
        Random rand = new Random();
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i][j] = rand.nextInt(10);
        return matrix;
    }

    public static int[][] multiplyParallelExecutor(int[][] A, int[][] B) throws InterruptedException, ExecutionException {
        int n = A.length;
        int[][] result = new int[n][n];

        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Void> completionService = new ExecutorCompletionService<>(executor);

        for (int i = 0; i < n; i++) {
            final int row = i;
            completionService.submit(() -> {
                for (int j = 0; j < n; j++) {
                    int sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += A[row][k] * B[k][j];
                    }
                    result[row][j] = sum;
                }
                return null;
            });
        }

        for (int i = 0; i < n; i++) {
            completionService.take().get();
        }

        executor.shutdown();
        return result;
    }

    public static void printMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long total = runtime.totalMemory();
        long free = runtime.freeMemory();
        long used = total - free;
        System.out.printf("Memory Usage: Used = %.2f MB, Free = %.2f MB, Total = %.2f MB%n",
                used / (1024.0 * 1024.0),
                free / (1024.0 * 1024.0),
                total / (1024.0 * 1024.0));
    }

    public static long getCpuTimeOfAllThreads(ExecutorService executor) {
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        if (!threadMXBean.isThreadCpuTimeSupported() || !threadMXBean.isThreadCpuTimeEnabled()) {
            return -1;
        }
        // We can't get exact threads from ExecutorService, so we get CPU time for all live threads
        long[] threadIds = threadMXBean.getAllThreadIds();
        long totalCpuTime = 0;
        for (long id : threadIds) {
            long time = threadMXBean.getThreadCpuTime(id);
            if (time != -1) {
                totalCpuTime += time;
            }
        }
        return totalCpuTime;
    }

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        int[] sizes = {500, 1000, 2000};  // Adjust sizes as you want

        for (int size : sizes) {
            System.out.println("Matrix size: " + size + "x" + size);

            int[][] A = generateRandomMatrix(size);
            int[][] B = generateRandomMatrix(size);

            // Warm-up run (optional)
            multiplyParallelExecutor(A, B);

            // Garbage collect before measuring
            System.gc();
            Thread.sleep(100);

            Runtime runtime = Runtime.getRuntime();
            long beforeUsedMem = runtime.totalMemory() - runtime.freeMemory();
            long beforeCpuTime = getCpuTimeOfAllThreads(null);
            long startTime = System.currentTimeMillis();

            multiplyParallelExecutor(A, B);

            long endTime = System.currentTimeMillis();
            long afterUsedMem = runtime.totalMemory() - runtime.freeMemory();
            long afterCpuTime = getCpuTimeOfAllThreads(null);

            System.out.println("Execution time: " + (endTime - startTime) + " ms");

            System.out.printf("Memory used during operation: %.2f MB%n", (afterUsedMem - beforeUsedMem) / (1024.0 * 1024.0));

            if (beforeCpuTime != -1 && afterCpuTime != -1) {
                long cpuTimeNano = afterCpuTime - beforeCpuTime;
                System.out.printf("CPU time used by threads: %.2f ms%n", cpuTimeNano / 1_000_000.0);
            } else {
                System.out.println("CPU time measurement not supported.");
            }

            printMemoryUsage();
            System.out.println("---------------------------------------------------\n");
        }
    }
}
