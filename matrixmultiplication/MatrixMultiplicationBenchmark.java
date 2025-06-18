package matrixmultiplication;

import java.util.*;
import java.io.*;

public class MatrixMultiplicationBenchmark {
    private static final int[] SIZES = {500, 1000, 2000}; // Add 5000, 10000 if you have enough RAM
    private static final int[] THREAD_COUNTS = {4, 8, 16};
    private static final double TOLERANCE = 1e-6;
    private static final boolean OUTPUT_CSV = true; // Set true to write CSV

    public static void main(String[] args) throws IOException {
        List<String[]> results = new ArrayList<>();
        results.add(new String[]{"Implementation", "MatrixSize", "Threads", "Time(ms)", "Memory(MB)", "Correct"});

        System.out.println("Java: " + System.getProperty("java.version") + ", Cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("==== Matrix Multiplication Benchmark ====");

        for (int size : SIZES) {
            System.out.println("\n===== Matrix Size: " + size + " x " + size + " =====");
            double[][] matrixA = MatrixUtils.generateRandomMatrix(size, size);
            double[][] matrixB = MatrixUtils.generateRandomMatrix(size, size);

            // Sequential baseline
            MatrixMultiplier sequential = new SequentialMatrixMultiplier();
            BenchmarkResult seqResult = runAndMeasure(sequential, matrixA, matrixB, -1);
            results.add(seqResult.toRow(size));
            System.out.println(seqResult);
            double[][] baseline = seqResult.result;

            for (int threads : THREAD_COUNTS) {
                // Concurrent
                MatrixMultiplier concurrent = new ConcurrentMatrixMultiplier(threads);
                BenchmarkResult concResult = runAndMeasure(concurrent, matrixA, matrixB, threads, baseline);
                results.add(concResult.toRow(size));
                System.out.println(concResult);

                // ThreadPool
                MatrixMultiplier threadPool = new ThreadPoolMatrixMultiplier(threads);
                BenchmarkResult tpResult = runAndMeasure(threadPool, matrixA, matrixB, threads, baseline);
                results.add(tpResult.toRow(size));
                System.out.println(tpResult);

                // ForkJoin
                MatrixMultiplier forkJoin = new ForkJoinMatrixMultiplier(threads);
                BenchmarkResult fjResult = runAndMeasure(forkJoin, matrixA, matrixB, threads, baseline);
                results.add(fjResult.toRow(size));
                System.out.println(fjResult);

                // BlockedParallel
                MatrixMultiplier blocked = new BlockedParallelMatrixMultiplier(threads, 64);
                BenchmarkResult bpResult = runAndMeasure(blocked, matrixA, matrixB, threads, baseline);
                results.add(bpResult.toRow(size));
                System.out.println(bpResult);
            }

            // ParallelStream (uses common pool, so no thread count param)
            MatrixMultiplier parallelStream = new ParallelStreamMatrixMultiplier();
            BenchmarkResult psResult = runAndMeasure(parallelStream, matrixA, matrixB, -1, baseline);
            results.add(psResult.toRow(size));
            System.out.println(psResult);
        }

        // Print summary table
        System.out.println("\n==== Summary Table ====");
        for (String[] row : results) {
            System.out.println(String.join(", ", row));
        }

        // Optionally write to CSV
        if (OUTPUT_CSV) {
            try (PrintWriter pw = new PrintWriter("matrix_benchmark_results.csv")) {
                for (String[] row : results) {
                    pw.println(String.join(",", row));
                }
            }
            System.out.println("Results written to matrix_benchmark_results.csv");
        }

        System.out.println("\nTip: For deeper profiling, use JVisualVM or Java Flight Recorder.");
    }

    private static BenchmarkResult runAndMeasure(MatrixMultiplier multiplier, double[][] A, double[][] B, int threads) {
        return runAndMeasure(multiplier, A, B, threads, null);
    }

    private static BenchmarkResult runAndMeasure(MatrixMultiplier multiplier, double[][] A, double[][] B, int threads, double[][] baseline) {
        System.gc();
        long memBefore = usedMemory();
        long start = System.nanoTime();
        double[][] result = multiplier.multiply(A, B);
        long end = System.nanoTime();
        long memAfter = usedMemory();
        double timeMs = (end - start) / 1e6;
        double memMB = (memAfter - memBefore) / (1024.0 * 1024.0);
        boolean correct = true;
        if (baseline != null) {
            correct = MatrixUtils.areMatricesEqual(baseline, result, TOLERANCE);
        }
        return new BenchmarkResult(multiplier.getName(), threads, timeMs, memMB, correct, result);
    }

    private static long usedMemory() {
        Runtime rt = Runtime.getRuntime();
        return rt.totalMemory() - rt.freeMemory();
    }

    private static class BenchmarkResult {
        String name;
        int threads;
        double timeMs;
        double memMB;
        boolean correct;
        double[][] result;
        BenchmarkResult(String name, int threads, double timeMs, double memMB, boolean correct, double[][] result) {
            this.name = name;
            this.threads = threads;
            this.timeMs = timeMs;
            this.memMB = memMB;
            this.correct = correct;
            this.result = result;
        }
        String[] toRow(int size) {
            return new String[]{name, size + "x" + size, threads == -1 ? "-" : String.valueOf(threads),
                    String.format("%.2f", timeMs), String.format("%.2f", memMB), correct ? "YES" : "NO"};
        }
        @Override
        public String toString() {
            return String.format("%-35s | Size: %6s | Threads: %3s | Time: %8.2f ms | Mem: %8.2f MB | Correct: %s",
                    name, result.length + "x" + result[0].length, threads == -1 ? "-" : threads, timeMs, memMB, correct ? "YES" : "NO");
        }
    }
} 