package com.excilys.gradureau.dl4jtest;

import com.excilys.gradureau.dl4jtest.utilities.DataUtilities;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class DiscoverDl4jApi {

    public static void main(String... args) throws Exception {
        MnistImagePipelineExample.main(args);
    }

    static class MnistImagePipelineExample {
        private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExample.class);


        /**
         * Location to save and extract the training/testing data
         */
        public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.home"), "dl4j_Mnist/");

        public static void main(String[] args) throws Exception {
            int height = 800;
            int width = 800;
            int channels = 1;
            int rngseed = 123;
            Random randNumGen = new Random(rngseed);
            int batchSize = 128;
            int outputNum = 26;
            int numEpochs = 1;

            // Define the File Paths
            File trainData = new File(DATA_PATH + "/mnist_png/training");
            File testData = new File(DATA_PATH + "/mnist_png/testing");


            // Define the FileSplit(PATH, ALLOWED FORMATS,random)
            FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

            // Extract the parent path as the image label
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

            ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

            // Initialize the record reader
            // add a listener, to extract the name
            recordReader.initialize(train);

            // The LogRecordListener will log the path of each image read
            // used here for information purposes,
            // If the whole dataset was ingested this would place 60,000
            // lines in our logs
            // It will show up in the output with this format
            // o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png
            recordReader.setListeners(new LogRecordListener());


            // DataSet Iterator
            DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

            // Scale pixel values to 0-1
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);

            // Build Our Neural Network
            log.info("BUILD MODEL");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(rngseed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Nesterovs(0.006, 0.9))
                    .l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(height * width)
                            .nOut(100)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(100)
                            .nOut(outputNum)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .pretrain(false).backprop(true)
                    .setInputType(InputType.convolutional(height, width, channels))
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            // The Score iteration Listener will log
            // output to show how well the network is training
            model.setListeners(new ScoreIterationListener(10));

            log.info("TRAIN MODEL");
            for (int i = 0; i < numEpochs; i++) {
                model.fit(dataIter);
            }

            log.info("SAVE TRAINED MODEL");
            // Where to save model
            File locationToSave = new File(DATA_PATH + "trained_mnist_model.zip");

            // boolean save Updater
            boolean saveUpdater = false;

            // ModelSerializer needs modelname, saveUpdater, Location
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        }
    }
}


