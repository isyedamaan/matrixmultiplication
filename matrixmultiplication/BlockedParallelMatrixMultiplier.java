package matrixmultiplication;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class BlockedParallelMatrixMultiplier implements MatrixMultiplier {
    private int parallelism;
    private int blockSize;

    public BlockedParallelMatrixMultiplier(int parallelism, int blockSize) {
        this.parallelism = parallelism;
        this.blockSize = blockSize;
    }

    public BlockedParallelMatrixMultiplier() {
        this(Runtime.getRuntime().availableProcessors(), 64);
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
        ForkJoinPool pool = new ForkJoinPool(parallelism);
        pool.invoke(new BlockMultiplyTask(matrixA, matrixB, result, 0, rowsA, 0, colsB));
        pool.shutdown();
        return result;
    }

    private class BlockMultiplyTask extends RecursiveAction {
        private final double[][] matrixA, matrixB, result;
        private final int rowStart, rowEnd, colStart, colEnd;

        BlockMultiplyTask(double[][] matrixA, double[][] matrixB, double[][] result,
                          int rowStart, int rowEnd, int colStart, int colEnd) {
            this.matrixA = matrixA;
            this.matrixB = matrixB;
            this.result = result;
            this.rowStart = rowStart;
            this.rowEnd = rowEnd;
            this.colStart = colStart;
            this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            int rowBlock = rowEnd - rowStart;
            int colBlock = colEnd - colStart;
            if (rowBlock <= blockSize && colBlock <= blockSize) {
                int colsA = matrixA[0].length;
                for (int i = rowStart; i < rowEnd; i++) {
                    for (int j = colStart; j < colEnd; j++) {
                        double sum = 0;
                        for (int k = 0; k < colsA; k++) {
                            sum += matrixA[i][k] * matrixB[k][j];
                        }
                        result[i][j] = sum;
                    }
                }
            } else if (rowBlock >= colBlock) {
                int midRow = (rowStart + rowEnd) / 2;
                invokeAll(
                    new BlockMultiplyTask(matrixA, matrixB, result, rowStart, midRow, colStart, colEnd),
                    new BlockMultiplyTask(matrixA, matrixB, result, midRow, rowEnd, colStart, colEnd)
                );
            } else {
                int midCol = (colStart + colEnd) / 2;
                invokeAll(
                    new BlockMultiplyTask(matrixA, matrixB, result, rowStart, rowEnd, colStart, midCol),
                    new BlockMultiplyTask(matrixA, matrixB, result, rowStart, rowEnd, midCol, colEnd)
                );
            }
        }
    }

    @Override
    public String getName() {
        return "BlockedParallel (ForkJoin, blockSize=" + blockSize + ", threads=" + parallelism + ")";
    }
} 