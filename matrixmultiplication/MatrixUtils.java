/**
 * MatrixUtils.java
 * Utility class for matrix operations such as generation and comparison
 */
package matrixmultiplication;

import java.util.Random;

public class MatrixUtils {
    
    /**
     * Generates a random matrix with the specified dimensions
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @return A matrix filled with random values between 0 and 10
     */
    public static double[][] generateRandomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        Random random = new Random();
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextDouble() * 10;
            }
        }
        
        return matrix;
    }
    
    /**
     * Compares two matrices for equality within a given tolerance
     * 
     * @param matrixA First matrix
     * @param matrixB Second matrix
     * @param tolerance Maximum allowed difference between elements
     * @return true if matrices are equal within tolerance, false otherwise
     */
    public static boolean areMatricesEqual(double[][] matrixA, double[][] matrixB, double tolerance) {
        if (matrixA.length != matrixB.length || matrixA[0].length != matrixB[0].length) {
            return false;
        }
        
        for (int i = 0; i < matrixA.length; i++) {
            for (int j = 0; j < matrixA[0].length; j++) {
                if (Math.abs(matrixA[i][j] - matrixB[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * Prints a matrix to console (for debugging)
     * 
     * @param matrix The matrix to print
     * @param name Name to display for the matrix
     * @param maxRows Maximum number of rows to print (to avoid flooding console)
     * @param maxCols Maximum number of columns to print
     */
    public static void printMatrix(double[][] matrix, String name, int maxRows, int maxCols) {
        int rows = Math.min(matrix.length, maxRows);
        int cols = Math.min(matrix[0].length, maxCols);
        
        System.out.println("Matrix " + name + " (" + matrix.length + "x" + matrix[0].length + "):");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%8.2f ", matrix[i][j]);
            }
            if (cols < matrix[0].length) {
                System.out.print("...");
            }
            System.out.println();
        }
        if (rows < matrix.length) {
            System.out.println("...");
        }
        System.out.println();
    }
    
    /**
     * Verifies if a matrix multiplication result is correct by comparing with sequential implementation
     * 
     * @param matrixA First input matrix
     * @param matrixB Second input matrix
     * @param result Result matrix to verify
     * @param tolerance Maximum allowed difference
     * @return true if result is correct, false otherwise
     */
    public static boolean verifyMatrixMultiplication(double[][] matrixA, double[][] matrixB, 
                                                    double[][] result, double tolerance) {
        // Calculate expected result using sequential algorithm
        SequentialMatrixMultiplier seqMultiplier = new SequentialMatrixMultiplier();
        double[][] expected = seqMultiplier.multiply(matrixA, matrixB);
        
        // Compare with provided result
        return areMatricesEqual(expected, result, tolerance);
    }
}