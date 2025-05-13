/**
 * MatrixMultiplier.java
 * Interface for all matrix multiplication implementations
 */
package matrixmultiplication;

public interface MatrixMultiplier {
    
    /**
     * Multiplies two matrices
     * 
     * @param matrixA First matrix
     * @param matrixB Second matrix
     * @return Result of multiplication
     */
    double[][] multiply(double[][] matrixA, double[][] matrixB);
    
    /**
     * Returns a descriptive name for the implementation
     * 
     * @return Name of the implementation
     */
    String getName();
}