/**
 * SequentialMatrixMultiplier.java
 * Implements the sequential version of matrix multiplication algorithm
 */
package matrixmultiplication;

public class SequentialMatrixMultiplier implements MatrixMultiplier {
    
    /**
     * Multiplies two matrices using the standard sequential algorithm
     * 
     * @param matrixA First matrix
     * @param matrixB Second matrix
     * @return Result of multiplication
     * @throws IllegalArgumentException if matrices cannot be multiplied
     */
    @Override
    public double[][] multiply(double[][] matrixA, double[][] matrixB) {
        int rowsA = matrixA.length;
        int colsA = matrixA[0].length;
        int rowsB = matrixB.length;
        int colsB = matrixB[0].length;
        
        // Check if matrices can be multiplied
        if (colsA != rowsB) {
            throw new IllegalArgumentException(
                "Matrix dimensions incompatible for multiplication: " +
                rowsA + "x" + colsA + " and " + rowsB + "x" + colsB
            );
        }
        
        // Initialize result matrix
        double[][] result = new double[rowsA][colsB];
        
        // Perform the multiplication
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                double sum = 0;
                for (int k = 0; k < colsA; k++) {
                    sum += matrixA[i][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        }
        
        return result;
    }
    
    @Override
    public String getName() {
        return "Sequential";
    }
}