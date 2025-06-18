package matrixmultiplication;

import java.util.stream.IntStream;

public class ParallelStreamMatrixMultiplier implements MatrixMultiplier {
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

        IntStream.range(0, rowsA).parallel().forEach(i -> {
            for (int j = 0; j < colsB; j++) {
                double sum = 0;
                for (int k = 0; k < colsA; k++) {
                    sum += matrixA[i][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        });

        return result;
    }

    @Override
    public String getName() {
        return "ParallelStream";
    }
} 