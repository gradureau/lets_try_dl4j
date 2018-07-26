package com.excilys.gradureau.dl4jtest;

import com.excilys.gradureau.dl4jtest.utilities.DataUtilities;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
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
         * Data URL for downloading
         */
        public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

        /**
         * Location to save and extract the training/testing data
         */
        public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.home"), "dl4j_Mnist/");

        public static void main(String[] args) throws Exception {
    /*
    image information
    28 * 28 grayscale
    grayscale implies single channel
    */
            int height = 800;
            int width = 800;
            int channels = 1;
            int rngseed = 123;
            Random randNumGen = new Random(rngseed);
            int batchSize = 30;
            int outputNum = 20;

            // Define the File Paths
            File trainData = new File(DATA_PATH + "/mnist_png/training");

            // Define the FileSplit(PATH, ALLOWED FORMATS,random)
            FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

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

            // In production you would loop through all the data
            // in this example the loop is just through 3
            // images for demonstration purposes
            for (int i = 1; i < 3; i++) {
                DataSet ds = dataIter.next();
                log.info(ds.toString());
                log.info(dataIter.getLabels().toString());
            }


        }
    }
}

