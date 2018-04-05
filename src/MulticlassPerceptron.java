import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
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
    private double[][] weights;
    private int numWeightUpdates = 0;
    private boolean debug = false;

    public MulticlassPerceptron(String[] options) {
        inputFileName = options[0] ;   // input file name
        numTrainingEpochs = Integer.parseInt(options[1]);   // number of training epochs
    }

    public void buildClassifier(Instances instances) throws Exception {
        data = new Instances(instances);
        int numClasses = data.numClasses();
        int numAttributes = data.numAttributes(); // Includes class attribute
        int numInst = data.numInstances();

        // Create a weight vector for each class with size equal to number of attributes plus a bias with each weight initialized to zero
        weights = new double[numClasses][numAttributes];

        Instance inst;
        int predictedClass, correctClass;
        for(int k = 0; k < numTrainingEpochs; k++) {
            System.out.print("Epoch\t" + (k+1) + ": ");

            for(int i = 0; i < numInst; i++) {
                inst =  data.get(i);
                predictedClass = predict(inst);

                if(debug) {
                    System.out.println("\nDEBUG~~~~ predictedClass = " + predictedClass + ", correctClass = " + inst.classValue());
                }

                // Update weights if predicted class is wrong
                correctClass = (int) inst.classValue();
                if (predictedClass != correctClass) {
                    System.out.print("0");

                    if(debug) {
                        System.out.println("\nDEBUG~~~~ Wrong Prediction => Update Weights");
                        System.out.println("DEBUG~~~~ Old Weights:");
                        System.out.print(appendWeights(new StringBuilder()).toString());
                    }

                    numWeightUpdates++;
                    for(int j = 0; j < numAttributes; j++) {
                        weights[predictedClass][j] -= inst.value(j);
                        weights[correctClass][j] += inst.value(j);
                    }

                    if(debug) {
                        System.out.println("\nDEBUG~~~~ New Weights");
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

    @Override
    public String toString() {
        String str = "Source File: " + inputFileName + "\nTraining epochs: " + numTrainingEpochs + "\nTotal # weight updates = " + numWeightUpdates + "\n\nFinal weights:\n\n";
        StringBuilder sb = new StringBuilder(str);
        sb = appendWeights(sb);
        return sb.toString();
    }

    // returns the class probabilities array of the prediction for the given weka.core.Instance object.
    // If your classifier handles nominal class attributes, then you need to override this method.
    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] result = new double[ data.numClasses() ];
        result[ predict(instance) ] = 1;
        return result;
    }

    // Required concrete implementation
    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    // Required concrete implementation
    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }

    private double[][] convertInstancesToMatrix(Instances instances) {
        int numInst = instances.numInstances();
        int numAttributes = instances.numAttributes();
        double[][] result = new double[numInst][numAttributes];
        Instance inst;

        for(int i = 0; i < numInst; i++) {
            inst = data.get(i);

            for (int j = 0; j < numAttributes - 1; j++) {
                result[i][j] = inst.value(j);
            }

            // Set the bias
            result[i][numAttributes - 1] = 1;
        }

        return result;
    }

    private int computeActivation(double[] w, Instance inst) {
        double sum = 0;
        double feature = 0;

        if(debug) {
            System.out.print("W * F = ");
        }

        for ( int i = 0; i < w.length; i++) {
            feature = i == w.length-1 ? 1 : inst.value(i);
            sum += w[i]*feature;

            if(debug) {
                System.out.print(w[i] + " * " + inst.value(i));
                if(i < w.length-1) {
                    System.out.print(" + ");
                }
            }
        }

        if(debug) {
            System.out.print(" = " + sum);
        }

        return sum < 0 ? -1 : 1;
    }

    private int predict(Instance inst) {
        int maxActivation = Integer.MIN_VALUE;
        int predictedClass = 0;
        int currentActivation = 0;

        for(int i = 0; i < inst.numClasses(); i++) {
            if(debug)
                System.out.print("\nDEBUG~~~~ Class " + i + ": ");

            currentActivation = computeActivation(weights[i], inst);

            if(debug) {
                System.out.print(" => Activation: " + currentActivation + ", maxActivation: " + maxActivation);
            }

            // Keep track of max activation
            if (currentActivation > maxActivation) {
                maxActivation = currentActivation;
                predictedClass = i;

                if(debug)
                    System.out.print("\nDEBUG~~~~ Updating Predicted Class to " + i);
            }
        }

        return predictedClass;
    }

    private StringBuilder appendWeights(StringBuilder sb) {
        for(int i = 0; i < weights.length; i++) {
            sb.append("Class ").append(i).append(" weights:\t");

            for(double weight: weights[i]) {
                sb.append(weight).append(" ");
            }
            sb.append("\n");
        }

        return sb;
    }
}
