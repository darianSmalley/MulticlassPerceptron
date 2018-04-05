import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * University of Central Florida
 * CAP4630 Artifical Intelligence
 * Multi-Class Vector Perceptron Classifier Class
 * Author: Darian Smalley
 */
public class MulticlassPerceptron implements weka.classifiers.Classifier {
    private Instances data;
    private String inputFileName;
    private int numTrainingEpochs;
    private int numWeightUpdates = 0;
    private int bias = 1;
    private double[][] weights;
    private boolean debug = false;

    public MulticlassPerceptron(String[] options) {
        inputFileName = options[0] ;   // input file name
        numTrainingEpochs = Integer.parseInt(options[1]);   // number of training epochs
    }

    public void buildClassifier(Instances instances) throws Exception {
        printHeader();

        data = new Instances(instances);
        double correction;
        int predictedClass, correctClass, numAttributes = data.numAttributes(); // Includes class attribute

        // Create a weight vector for each class with size equal to number of attributes plus a bias with each weight initialized to zero
        weights = new double[data.numClasses()][numAttributes];


        for(int i = 0; i < numTrainingEpochs; i++) {
            System.out.print("Epoch\t" + (i+1) + ": ");

            for(Instance inst : instances) {
                predictedClass = predict(inst);
                correctClass = (int) inst.classValue();

                if(debug) {
                    System.out.println("\n\tDEBUG\t predictedClass = " + predictedClass + ", correctClass = " + inst.classValue());
                }

                // Incorrect prediction, update weights
                if (predictedClass != correctClass) {
                    System.out.print("0");

                    if(debug) {
                        System.out.println("\n\tDEBUG\t Wrong Prediction. Old Weights:");
                        System.out.print(appendWeights(new StringBuilder()).toString());
                    }

                    numWeightUpdates++;
                    for(int j = 0; j < numAttributes; j++) {
                        correction = (j == numAttributes - 1) ? bias : inst.value(j);
                        weights[predictedClass][j] -= correction;
                        weights[correctClass][j] += correction;
                    }

                    if(debug) {
                        System.out.println("\n\tDEBUG\t New Weights");
                        System.out.print(appendWeights(new StringBuilder()).toString());
                    }
                } else {
                    // Correct prediction
                    System.out.print("1");
                }
            }

            System.out.println();
        }
    }

    private int computeActivation(double[] w, Instance inst) {
        double sum = 0;

        if(debug) {
            System.out.print("W * F = ");
        }

        for ( int i = 0; i < w.length-1; i++) {
            sum += w[i] * inst.value(i);

            if(debug) {
                System.out.print(w[i] + " * " + inst.value(i));
                if(i < w.length-1) {
                    System.out.print(" + ");
                }
            }
        }

        // last element is bias weight
        sum += w[w.length-1] * bias;

        if(debug) {
            System.out.print(" = " + sum);
        }

        return sum < 0 ? -1 : 1;
    }

    private int predict(Instance inst) {
        int maxActivation = Integer.MIN_VALUE;
        int predictedClass = 0;
        int currentActivation;

        for(int i = 0; i < inst.numClasses(); i++) {
            if(debug)
                System.out.print("\n\tDEBUG\t Class " + i + ": ");

            currentActivation = computeActivation(weights[i], inst);

            if(debug) {
                System.out.print(" => Activation: " + currentActivation + ", maxActivation: " + maxActivation);
            }

            // Keep track of max activation
            if (currentActivation > maxActivation) {
                maxActivation = currentActivation;
                predictedClass = i;

                if(debug)
                    System.out.print("\n\tDEBUG\t Updating Predicted Class to " + i);
            }
        }

        return predictedClass;
    }

    private StringBuilder appendWeights(StringBuilder sb) {
        for(int i = 0; i < weights.length; i++) {
            sb.append("Class ").append(i).append(" weights:\t");

            for(double weight: weights[i]) {
                sb.append(String.format(java.util.Locale.US,"%.3f", weight)).append(" ");
            }

            if (i < weights.length-1) {
                sb.append("\n");
            }
        }

        return sb;
    }

    private void printHeader() {
        System.out.println("University of Central Florida");
        System.out.println("CAP4630 Artifical Intelligence - Spring 2018");
        System.out.println("Multi-Class Perceptron Classifier by Darian Smalley\n");
    }

    @Override
    public String toString() {
        String str = "Source File: " + inputFileName + "\nTraining epochs: " + numTrainingEpochs + "\nTotal # weight updates = " + numWeightUpdates + "\n\nFinal weights:\n\n";
        StringBuilder sb = new StringBuilder(str);
        sb = appendWeights(sb);
        return sb.toString();
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] result = new double[ data.numClasses() ];
        result[ predict(instance) ] = 1;
        return result;
    }

    /**
     * Required concrete implementation
     */
    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    /**
     * Required concrete implementation
     */
    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }
}
