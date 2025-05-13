/**
 * ConcurrentMatrixMultiplier.java
 * Implements matrix multiplication using manually created threads
 */
package matrixmultiplication;

public class ConcurrentMatrixMultiplier implements MatrixMultiplier {

  private int numThreads;

  /**
   * Constructor with specified number of threads
   *
   * @param numThreads Number of threads to use for multiplication
   */
  public ConcurrentMatrixMultiplier(int numThreads) {
    this.numThreads = numThreads;
  }

  /**
   * Default constructor uses available processors count
   */
  public ConcurrentMatrixMultiplier() {
    this(Runtime.getRuntime().availableProcessors());
    System.out.println(
        "Concurrent Approach: using default number of threads - " +
        Runtime.getRuntime().availableProcessors());
  }

  /**
   * Multiplies two matrices using multiple threads
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
          "Matrix dimensions incompatible for multiplication: " + rowsA + "x" +
          colsA + " and " + rowsB + "x" + colsB);
    }

    // Initialize result matrix
    double[][] result = new double[rowsA][colsB];

    // Create and start threads
    Thread[] threads = new Thread[numThreads];
    int rowsPerThread = Math.max(1, rowsA / numThreads);

    for (int t = 0; t < numThreads; t++) {
      final int threadId = t;
      final int startRow = threadId * rowsPerThread;
      final int endRow = (threadId == numThreads - 1)
                             ? rowsA
                             : Math.min(startRow + rowsPerThread, rowsA);

      // Skip thread creation if no rows to process
      if (startRow >= rowsA)
        continue;

      threads[t] = new Thread(() -> {
        // Each thread computes its portion of the result matrix
        for (int i = startRow; i < endRow; i++) {
          for (int j = 0; j < colsB; j++) {
            double sum = 0;
            for (int k = 0; k < colsA; k++) {
              sum += matrixA[i][k] * matrixB[k][j];
            }
            result[i][j] = sum;
          }
        }
      });

      threads[t].start();
    }

    // Wait for all threads to complete
    try {
      for (int t = 0; t < numThreads; t++) {
        if (threads[t] != null) {
          threads[t].join();
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("Matrix multiplication interrupted", e);
    }

    return result;
  }

  @Override
  public String getName() {
    return "Concurrent (" + numThreads + " threads)";
  }
}
